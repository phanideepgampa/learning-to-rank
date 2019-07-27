import os
import argparse
import logging
import random
import pickle
from collections import namedtuple
import time
import traceback

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from lr_scheduler import ReduceLROnPlateau

import model
import evaluate
from data_preprocessing import PickleReader,BatchDataLoader,Context,Dataset
from bandit import ContextualBandit


np.set_printoptions(precision=4, suppress=True)

Config = namedtuple('parameters',
                    ['input_dim',
                     'dropout','highway','nn_layers','num_class','multi_class','mid_dim'])


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ml_loss_1(context,outputs):
    true_labels = np.zeros(len(context.labels))
    true_labels[context.labels>0]=1.0
    return F.binary_cross_entropy(outputs.view(-1),torch.tensor(true_labels).type(torch.float).cuda())
    
#def ml_loss_2(context,outputs):
#    return F.cross_entropy(outputs[1],torch.LongTensor(context.labels).cuda())

def train_model(args):
    print(args)
    print("generating config")
    config = Config(
        input_dim=args.input_dim,
        dropout=args.dropout,
        highway=args.highway,
        nn_layers = args.nn_layers,
        num_class = args.num_class,
        multi_class= args.multi_class,
        mid_dim = args.mid_dim
    )
    model_name = ".".join((args.model_file,
                           str(args.rl_baseline_method),args.sampling_method,
                           "gamma",str(args.gamma),
                           "beta",str(args.beta),
                           "batch",str(args.train_batch),
                           "lr",str(args.lr)+"-"+str(args.lr_sch),
                           "bsz", str(args.batch_size), 
                           "data", args.data_dir.split('/')[0]+args.data_dir.split('/')[1][-1],
                           "input_dim", str(config.input_dim),str(args.mid_dim),"test",
                           "dropout", str(args.dropout),
                           "max_num",str(args.max_num_of_ans),
                           "highway-3",str(args.highway),
                          # "multi_class",str(args.multi_class),
                           "nn_layers",str(args.nn_layers),
                           "reward_type",str(args.reward_type),
                           # "explore-0.25",
                           'ans'))

    log_name = ".".join(("log/model",
                           str(args.rl_baseline_method), args.sampling_method,
                           "gamma",str(args.gamma),
                           "beta",str(args.beta),
                            "batch",str(args.train_batch),
                           "lr",str(args.lr)+"-"+str(args.lr_sch),args.sampling_method,
                           "bsz", str(args.batch_size), 
                           "data", args.data_dir.split('/')[0]+args.data_dir.split('/')[1][-1],
                           "input_dim", str(config.input_dim),str(args.mid_dim),"test",
                           "dropout", str(args.dropout),
                           "max_num",str(args.max_num_of_ans),
                           "highway-3",str(args.highway),
    #                        "multi_class",str(args.multi_class),
                            "nn_layers",str(args.nn_layers),
                            "reward_type",str(args.reward_type),
                           # "explore-0.25",
                           'ans'))

    print("initialising data loader and RL learner")
    data_loader = PickleReader(args.data_dir)
    data = args.data_dir.split('/')[0]
    num_data = 0
    if data == "MQ2007":
        num_data = 339*3
    elif data == "MQ2008":
        num_data = 471
    else:
        assert(1==2)
    # init statistics
    reward_list = []
    loss_list =[]
    best_eval_reward = 0.
    model_save_name = model_name
    #if args.multi_class:
    #    ml_func = ml_loss_2
    #else:
    #    ml_func = ml_loss_1
    ml_func = ml_loss_1
    bandit = ContextualBandit(b=args.batch_size,rl_baseline_method=args.rl_baseline_method,sample_method=args.sampling_method,reward_type=args.reward_type)

    print("Loaded the Bandit")
 
    mcan_cb = model.MCAN_CB(config)

    print("Loaded the model")

    mcan_cb.cuda()

    if args.load_ext:
        model_name = args.model_file
        print("loading existing model%s" % model_name)
        mcan_cb = torch.load(model_name, map_location=lambda storage, loc: storage)
        mcan_cb.cuda()
        model_save_name=model_name
        log_name = "/".join(("log",model_name.split("/")[1])) 
        print("finish loading and evaluate model %s" % model_name)
        # evaluate.ext_model_eval(extract_net, vocab, args, eval_data="test")
        best_eval_reward = evaluate.ext_model_eval(mcan_cb,args, "test")[0]
    logging.basicConfig(filename='%s.log' % log_name,
                        level=logging.DEBUG, format='%(asctime)s %(levelname)-10s %(message)s')
    # Loss and Optimizer
    optimizer_ans = torch.optim.Adam([param for param in mcan_cb.parameters() if param.requires_grad == True ], lr=args.lr, betas=(args.beta, 0.999),weight_decay=1e-6)
    if args.lr_sch ==1:    
        scheduler = ReduceLROnPlateau(optimizer_ans, 'max',verbose=1,factor=0.9,patience=3,cooldown=3,min_lr=9e-5,epsilon=1e-6)
        if best_eval_reward:
            scheduler.step(best_eval_reward,0)
            print("init_scheduler")
    elif args.lr_sch ==2:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_ans,args.lr, args.lr_2, step_size_up=3*int(num_data/args.train_batch), step_size_down=2*int(num_data/args.train_batch), mode='exp_range', gamma=0.98,cycle_momentum=False)   
    print("starting training")
    start_time = time.time()
    n_step = 100
    gamma = args.gamma
    if num_data < 2000:

        n_val = int(num_data/(5*args.train_batch))
    else:
        n_val = int(num_data/(7*args.train_batch))
    with torch.autograd.set_detect_anomaly(True):
        for epoch in tqdm(range(args.epochs_ext),desc="epoch:"):
            train_iter = data_loader.chunked_data_reader("train", data_quota=args.train_example_quota)  #-1
            step_in_epoch = 0
            for dataset in train_iter:
                for step, contexts in tqdm(enumerate(BatchDataLoader(dataset, batch_size=args.train_batch,shuffle=True))):
                    try:
                        mcan_cb.train()
                        step_in_epoch += 1
                        loss=0.
                        reward=0.
                        for context in contexts:
                
                            q_a = torch.autograd.Variable(torch.from_numpy(context.features).type(torch.float)).cuda()
                            outputs = mcan_cb(q_a)

                            #if args.prt_inf and np.random.randint(0, 100) == 0:
                            #    prt = True
                            #else:
                            #    prt = False

                            loss_t, reward_t = bandit.train(outputs, context,
                                                        max_num_of_ans=args.max_num_of_ans,
                                                        prt=False)
                            #print(str(loss_t)+' '+str(len(a_len)))
                            
                            #    loss_t = loss_t.view(-1)
                            #true_labels = np.zeros(len(context.labels))
                            #true_labels[context.labels>0]=1.0
                            #ml_loss = F.binary_cross_entropy(outputs.view(-1),torch.tensor(true_labels).type(torch.float).cuda())
                            ml_loss = ml_func(context,outputs)
                            loss_e=((gamma*loss_t)+((1-gamma)*ml_loss))
                            loss_e.backward()
                            loss+=loss_e.item()
                            reward+=reward_t
                        loss = loss/args.train_batch
                        reward = reward/args.train_batch
                        #if prt:
                        #    print('Probabilities: ', outputs.squeeze().data.cpu().numpy())
                        #    print('-' * 80)

                        reward_list.append(reward)
                        loss_list.append(loss)
                        #if isinstance(loss, Variable):
                        #    loss.backward()

                        if step % 1 == 0:
                            # torch.nn.utils.clip_grad_norm_(mcan_cb.parameters(), 1)  # gradient clipping
                            optimizer_ans.step()
                            optimizer_ans.zero_grad()
                        if args.lr_sch==2:
                            scheduler.step()    
                        logging.info('Epoch %d Step %d Reward %.4f Loss %.4f' % (epoch, step_in_epoch, reward,loss))
                    except Exception as e:
                        print(e)
                        print(outputs)
                        print(ml_loss)
                        print(loss_t)
                        traceback.print_exc()

                    if (step_in_epoch) % n_step == 0 and step_in_epoch != 0:
                        logging.info('Epoch ' + str(epoch) + ' Step ' + str(step_in_epoch) +
                            ' reward: ' + str(np.mean(reward_list))+' loss: ' + str(np.mean(loss_list)))
                        reward_list = []
                        loss_list = []

                    if (step_in_epoch) % n_val == 0 and step_in_epoch != 0:
                        print("doing evaluation")
                        mcan_cb.eval()
                        eval_reward = evaluate.ext_model_eval(mcan_cb,args, "test")
                        
                        if  eval_reward[0] > best_eval_reward:
                            best_eval_reward = eval_reward[0]
                            print("saving model %s with eval_reward:" % model_save_name, eval_reward)
                            logging.debug("saving model"+str(model_save_name)+"with eval_reward:"+ str(eval_reward))
                            torch.save(mcan_cb, model_name)
                        print('epoch ' + str(epoch) + ' reward in validation: '
                            + str(eval_reward))
                        logging.debug('epoch ' + str(epoch) + ' reward in validation: '
                            + str(eval_reward))
                        logging.debug('time elapsed:'+str(time.time()-start_time))
            if args.lr_sch ==1:
                mcan_cb.eval()
                eval_reward = evaluate.ext_model_eval(mcan_cb,args, "test")
                scheduler.step(eval_reward[0],epoch)
    return mcan_cb

def main():
    seed_everything()
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='MQ2007/Fold/pickle_data/')
    parser.add_argument('--model_file', type=str, default='model/rank.model')
    parser.add_argument('--highway',action='store_true')
    parser.add_argument('--multi_class',action='store_true')
    parser.add_argument('--num_class',type=int,default=3)
    parser.add_argument('--beta',type=float,default = 0.9)
    parser.add_argument('--gamma',type=float,default=0.9)
    parser.add_argument('--lr_sch',type=int,default=0)
    parser.add_argument('--train_batch',type=int,default=1)
    parser.add_argument('--sampling_method',type=str,default = "herke")
    parser.add_argument('--max_num_of_ans',type=int,default=50)
    parser.add_argument('--epochs_ext', type=int, default=10)
    parser.add_argument('--load_ext', action='store_true')
    parser.add_argument('--input_dim', type=int, default=46)
    parser.add_argument('--mid_dim', type=int, default=46)
    parser.add_argument('--nn_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--lr_2', type=float, default=9e-5)
    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--rl_baseline_method', type=str, default="batch_avg",
                        help='greedy, global_avg, batch_avg, batch_med, or none')
    parser.add_argument('--rl_loss_method', type=int, default=2,
                        help='1 for computing 1-log on positive advantages,'
                             '0 for not computing 1-log on all advantages')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--reward_type', type=int, default=1)
    parser.add_argument('--train_example_quota', type=int, default=-1,
                        help='how many train example to train on: -1 means full train data')
    parser.add_argument('--prt_inf', action='store_true')

    args = parser.parse_args()


    torch.cuda.set_device(args.device)

    mcan_cb = train_model(args)


if __name__ == '__main__':
    main()
    
