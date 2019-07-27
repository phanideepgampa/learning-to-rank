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
from data_preprocessing import PickleReader,BatchDataLoader,Vocab,Context,Dataset
from bandit import ContextualBandit

"""
stats of TrecQa: Number of Answers for a Question
max ans:761         max positive ans: 51        max length of ans: 60
min ans:1           min positive ans: 1         min length of ans:1
mean ans:43.4       mean positve ans: 5.20      mean length of ans:27.7

"""
np.set_printoptions(precision=4, suppress=True)

Config = namedtuple('parameters',
                    ['vocab_size', 'embedding_dim',
                     'word_input_size','num_factors',
                     'LSTM_hidden_units','compression_type',
                     'pretrained_embedding', 'word2id', 'id2word',
                     'dropout','highway'])


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_model(args,vocab):
    print(args)
    print("generating config")
    config = Config(
        vocab_size=vocab.embedding.shape[0],
        embedding_dim=vocab.embedding.shape[1],
        word_input_size=vocab.embedding.shape[1]+12,
        LSTM_hidden_units= args.hidden,
        num_factors=args.num_factors,
        compression_type=args.compression_type,
        pretrained_embedding=vocab.embedding,
        word2id=vocab.word_to_index,
        id2word=vocab.index_to_word,
        dropout=args.dropout,
        highway=args.highway
    )
    model_name = ".".join((args.model_file,
                           str(args.rl_baseline_method),args.sampling_method,
                           "gamma",str(args.gamma),
                           "beta",str(args.beta),
                           "batch",str(args.train_batch),
                           "learning_rate",str(args.lr),
                           "compression",str(args.compression_type),
                           "bsz", str(args.batch_size), 
                           "data", args.data_dir.split('/')[0],
                           "emb", str(config.embedding_dim),
                           "dropout", str(args.dropout),
                           "max_num",str(args.max_num_of_ans),
                           "highway",str(args.highway),
                           'ans'))
    if args.compression_type == "FM":
        model_name=".".join((model_name,"num_factors",str(args.num_factors)))               
    print(model_name)

    log_name = ".".join(("log/model",
                           str(args.rl_baseline_method), args.sampling_method,
                           "gamma",str(args.gamma),
                           "beta",str(args.beta),
                            "batch",str(args.train_batch),
                           "lr",str(args.lr),args.sampling_method,
                           "compression",str(args.compression_type),
                           "bsz", str(args.batch_size), 
                           "data", args.data_dir.split('/')[0],
                           "emb", str(config.embedding_dim),
                           "dropout", str(args.dropout),
                           "max_num",str(args.max_num_of_ans),
                           "highway",str(args.highway),
                           'ans'))

    if args.compression_type == "FM":
        log_name=".".join((log_name,"num_factors",str(args.num_factors)))
    if args.cast_12:
        log_name+=".cast_12"
        model_name+=".cast_12"
    print("initialising data loader and RL learner")
    data_loader = PickleReader(args.data_dir)
    data = args.data_dir.split('/')[0]
    num_data = 0
    if data == "trec_qa":
        num_data = 1229
    elif data == "wiki_qa":
        num_data = 873
    elif data == "insurance_qa":
        num_data = 12887
    else:
        assert(1==2)
    # init statistics
    reward_list = []
    loss_list =[]
    best_eval_reward = 0.
    model_save_name = model_name

    bandit = ContextualBandit(b=args.batch_size,rl_baseline_method=args.rl_baseline_method,sample_method=args.sampling_method)

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
        best_eval_reward = evaluate.ext_model_eval(mcan_cb, vocab, args, "val")[0]
    print(mcan_cb.word_embedding.weight.requires_grad)
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
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_ans,args.lr, args.lr_2, step_size_up=3*int(num_data/args.train_batch), step_size_down=3*int(num_data/args.train_batch), mode='exp_range', gamma=0.98,cycle_momentum=False)   
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
                            

                            q,a,a_len,_ = model.word_to_tokens(context.question,context.answers,vocab)
                            # q = torch.autograd.Variable(q)
                            # a = torch.autograd.Variable(a)
                            q = torch.autograd.Variable(q).cuda()
                            a = torch.autograd.Variable(a).cuda()
                            #q.cuda()
                            #a.cuda()
                            a_len.cuda()
                            outputs = mcan_cb(q,a,a_len)

                            if args.prt_inf and np.random.randint(0, 100) == 0:
                                prt = True
                            else:
                                prt = False

                            loss_t, reward_t = bandit.train(outputs, context,
                                                        max_num_of_ans=args.max_num_of_ans,
                                                        #max_num_of_ans=len(a_len),
                                                        prt=prt)
                            #print(str(loss_t)+' '+str(len(a_len)))
                            
                            #    loss_t = loss_t.view(-1)
                            ml_loss = F.binary_cross_entropy(outputs.view(-1),torch.tensor(context.labels).type(torch.float).cuda())
                            
                            loss_e=((gamma*loss_t)+((1-gamma)*ml_loss))
                            loss_e.backward()
                            loss+=loss_e.item()
                            reward+=reward_t
                        loss = loss/args.train_batch
                        reward = reward/args.train_batch
                        if prt:
                            print('Probabilities: ', outputs.squeeze().data.cpu().numpy())
                            print('-' * 80)

                        reward_list.append(reward)
                        loss_list.append(loss)
                        #if isinstance(loss, Variable):
                        #    loss.backward()

                        if step % 1 == 0:
                            # torch.nn.utils.clip_grad_norm_(mcan_cb.parameters(), 1)  # gradient clipping
                            optimizer_ans.step()
                            optimizer_ans.zero_grad()
                        # print('Epoch %d Step %d Reward %.4f'%(epoch,step_in_epoch,reward))
                        if args.lr_sch==2:
                            scheduler.step()    
                        logging.info('Epoch %d Step %d Reward %.4f Loss %.4f' % (epoch, step_in_epoch, reward,loss))
                    except Exception as e:
                        print(e)
                        #print(loss)
                        #print(loss_e)
                        traceback.print_exc()

                    if (step_in_epoch) % n_step == 0 and step_in_epoch != 0:
                        logging.info('Epoch ' + str(epoch) + ' Step ' + str(step_in_epoch) +
                            ' reward: ' + str(np.mean(reward_list))+' loss: ' + str(np.mean(loss_list)))
                        reward_list = []
                        loss_list = []

                    if (step_in_epoch) % n_val == 0 and step_in_epoch != 0:
                        print("doing evaluation")
                        mcan_cb.eval()
                        #eval_reward = evaluate.ext_model_eval(mcan_cb, vocab, args, "test")
                        eval_reward = evaluate.ext_model_eval(mcan_cb, vocab, args, "val")
                        
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
                eval_reward = evaluate.ext_model_eval(mcan_cb, vocab, args, "val")
                #eval_reward = evaluate.ext_model_eval(mcan_cb, vocab, args, "test")
                scheduler.step(eval_reward[0],epoch)
    return mcan_cb

def main():
    seed_everything()
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_file', type=str, default='wiki_qa//vocab.300d.p')
    parser.add_argument('--data_dir', type=str, default='trec_qa/pickle_data/')
    parser.add_argument('--model_file', type=str, default='model/select.model')
    parser.add_argument('--highway',action='store_true')
    # parser.add_argument('--cast_12',action='store_true')
    parser.add_argument('--beta',type=float,default = 0.9)
    parser.add_argument('--gamma',type=float,default=0.9)
    parser.add_argument('--lr_sch',type=int,default=1)
    parser.add_argument('--max_num_of_ans',type=int,default=5)
    parser.add_argument('--train_batch',type=int,default=1)
    parser.add_argument('--sampling_method',type=str,default = "herke")
    parser.add_argument('--epochs_ext', type=int, default=10)
    parser.add_argument('--load_ext', action='store_true')
    parser.add_argument('--hidden', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--lr_2', type=float, default=8e-5)
    parser.add_argument('--num_factors',type=int,default=10,help='Number of factors in Factorization Machine')
    parser.add_argument('--compression_type',type=str,default="FM",help='Choose between 3 compression types:SM,NN,FM' )
    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--oracle_length', type=int, default=3,
                        help='-1 for giving actual oracle number of sentences'
                             'otherwise choose a fixed number of sentences')
    parser.add_argument('--rl_baseline_method', type=str, default="batch_avg",
                        help='greedy, global_avg, batch_avg, batch_med, or none')
    parser.add_argument('--rl_loss_method', type=int, default=2,
                        help='1 for computing 1-log on positive advantages,'
                             '0 for not computing 1-log on all advantages')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--train_example_quota', type=int, default=-1,
                        help='how many train example to train on: -1 means full train data')
    parser.add_argument('--prt_inf', action='store_true')

    args = parser.parse_args()


    torch.cuda.set_device(args.device)

    print('generate config')
    with open(args.vocab_file, "rb") as f:
        vocab = pickle.load(f)
    print(vocab)

    mcan_cb = train_model(args, vocab)


if __name__ == '__main__':
    main()
    
