import argparse
import time
from copy import deepcopy

import torch
import numpy as np
import pickle
import scipy.stats

import data_preprocessing
from data_preprocessing import Context,Dataset
import tempfile
from bandit import return_answer_index
from rank_metrics import average_precision,mean_reciprocal_rank,ndcg_at_k,dcg_at_k,precision_at_k

np.set_printoptions(precision=4, suppress=True)

def generate_reward(gold_index_list, answer_index_list):
    reward = 0
    ap=0
    reciprocal_rank=0
    answer_list =list(answer_index_list)
    size =len(answer_index_list)
    true = sum(gold_index_list>0)
    inp = np.zeros(size)
    for rank,val in enumerate(gold_index_list):
        if val and rank in answer_list:
            inp[answer_list.index(rank)]=val
    maxk = sum(inp>0)
    if true:
        ap=average_precision(inp)*(maxk/true)
    reciprocal_rank = mean_reciprocal_rank([inp])
    ndcg = ndcg_at_k(inp,min(10,size))
    dcg_five = dcg_at_k(inp,5)
    reward = (ap+reciprocal_rank+ndcg+dcg_five)/4
    ranks = [1,3,5,10]
    reward_tuple = [reward,ap,reciprocal_rank,ndcg,dcg_five]
    for r in ranks:
        reward_tuple.append(precision_at_k(inp,min(r,len(inp))))
    for r in ranks:
        reward_tuple.append(ndcg_at_k(inp,min(r,len(inp))))
    return reward_tuple



def reinforce_loss(probs, context,
                   max_num_of_ans=3):
    # sample sentences
    probs_numpy = probs.data.cpu().numpy()
    probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
    max_num_of_ans = min(len(probs_numpy), max_num_of_ans)  # max of sents# in doc and sents# in summary
    answer_index_list,_ = return_answer_index(probs_numpy,probs,sample_method="greedy",max_num_of_ans=len(probs_numpy))
    gold_index_list = context.labels
    reward = generate_reward(gold_index_list,answer_index_list)
    return reward,answer_index_list


def ext_model_eval(model,args,eval_data="test",trec_file="trecfile.txt"):
    print("loading data %s" % eval_data)

    model.eval()

    data_loader = data_preprocessing.PickleReader(args.data_dir)
    eval_rewards = []
    data_iter = data_loader.chunked_data_reader(eval_data)
    print("doing model evaluation on %s" % eval_data)
    trec_result=[]
    step_in_epoch=0
    for phase, dataset in enumerate(data_iter):
        for step, contexts in enumerate(data_preprocessing.BatchDataLoader(dataset, shuffle=False)):
            print("Done %2d chunck, %4d/%4d context\r" % (phase+1, step + 1, len(dataset)), end='')            
            context = contexts[0]
            q_a = torch.autograd.Variable(torch.from_numpy(context.features).type(torch.float)).cuda()
            outputs = model(q_a)
            if isinstance(outputs,list):
                outputs = outputs[0]
            reward,rank_list = reinforce_loss(outputs,context,max_num_of_ans=args.max_num_of_ans)
            eval_rewards.append(reward)
            step_in_epoch+=1
    avg_eval_r = np.mean(eval_rewards, axis=0)
    print('model reward in %s:' % (eval_data))
    print(avg_eval_r)
    return avg_eval_r

def statistical_tests(args):
    model_name = args.model_file.split(".")
    data_dir = args.data_dir.split("/")
    data=data_dir[0]
    folds = ['1','2','3','4','5']
    dropouts = ['4','3','3','3','3']
    mean_rewards=[]
    eval_rewards =[]
    model_name1=""
    for ind,fold in enumerate(folds):
        model_name = args.model_file.split(".")
        model_name[model_name.index("data")+1]= data+fold
        model_name[model_name.index("dropout")+2] = dropouts[ind]
        #if dropouts[ind] == '3':
        #    model_name[model_name.index("max_num")+1] = '30'
        data_dir[1]="Fold"+fold
        mcan_cb = torch.load(".".join(model_name), map_location=lambda storage, loc: storage)
        mcan_cb.cuda()
        print("finish loading  model %s" % ".".join(model_name))
        mcan_cb.eval()

        data_loader = data_preprocessing.PickleReader("/".join(data_dir))
        data_iter = data_loader.chunked_data_reader("test")
        trec_result=[]
        step_in_epoch=0
        mean_arr =[]
        for phase, dataset in enumerate(data_iter):
            for step, contexts in enumerate(data_preprocessing.BatchDataLoader(dataset, shuffle=False)):
                print("Done %2d chunck, %4d/%4d context\r" % (phase+1, step + 1, len(dataset)), end='')            
                context = contexts[0]
                q_a = torch.autograd.Variable(torch.from_numpy(context.features).type(torch.float)).cuda()
                outputs = mcan_cb(q_a)
                if isinstance(outputs,list):
                    outputs = outputs[0]
                reward,rank_list = reinforce_loss(outputs,context,max_num_of_ans=args.max_num_of_ans)
                eval_rewards.append(reward)
                mean_arr.append(reward)
                step_in_epoch+=1
        if len(mean_rewards):
            mean_rewards+=np.mean(mean_arr,axis=0)
        else:
            mean_rewards = np.mean(mean_arr,axis=0)
    eval_rewards = np.array(eval_rewards).transpose()
    metrics = ['P@1','P@3','P@5','P@10','NDCG@1','NDCG@3','NDCG@5','NDCG@10','MAP','RR@52']
    baseline_models= ['ada.scores.p','listnet.scores.p','cascent.scores.p','lambda.scores.p']
    results = open("statistical_result.txt","w")
    results.write("\t".join(["test"]+metrics)+"\n")
    for baseline in baseline_models:
        results.write("\n"+baseline+"\n")
        t_test=[]
        wilcoxon=[]
        with open(baseline,"rb+") as f:
            scores=pickle.load(f)
        for num,metric in enumerate(metrics):
            t_test.append(str(scipy.stats.ttest_rel(eval_rewards[num],scores[num])[-1]))
            wilcoxon.append(str(scipy.stats.wilcoxon(eval_rewards[num],scores[num])[-1]))

        results.write("\t".join(["t-test"]+t_test)+"\n")
        results.write("\t".join(["wilcoxon"]+wilcoxon)+"\n")
    mean_rewards/=5.
    results.write(np.array2string(mean_rewards,precision=4,separator="\t"))
    results.close()

if __name__ == '__main__':
    import pickle

    torch.manual_seed(1234)
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='trec_qa/pickle_data/')
    parser.add_argument('--model_file', type=str, default='model/select.ans')
    parser.add_argument('--trec_file',type=str,default="trec_file.txt")
    parser.add_argument('--statistics',action="store_true")
    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--rl_baseline_method', type=str, default="greedy",
                        help='greedy, global_avg,batch_avg,or none')
    parser.add_argument('--length_limit', type=int, default=-1,
                        help='length limit output')
    parser.add_argument('--max_num_of_ans', type=int, default=-1,
                        help='max ans to rank')

    args = parser.parse_args()
    #assert(args.max_num_of_ans!= -1)

    torch.cuda.set_device(args.device)
    if args.statistics:
        statistical_tests(args)
        exit() 

    model_name = args.model_file.split('.')
    maxk=0
    maxk = int(model_name[model_name.index('max_num')+1])
    assert(maxk!=0)
    args.max_num_of_ans = maxk 
    assert(args.data_dir[-1]==model_name[model_name.index('data')+1][-1])

    print("loading existing model%s" % args.model_file)
    mcan_cb = torch.load(args.model_file, map_location=lambda storage, loc: storage)
    mcan_cb.cuda()
    print("finish loading and evaluate model %s" % args.model_file)

    start_time = time.time()
    ext_model_eval(mcan_cb,args,eval_data="test",trec_file= args.trec_file)
    print('Test time:', time.time() - start_time)
