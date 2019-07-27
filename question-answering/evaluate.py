import argparse
import time

import torch
import numpy as np

import data_preprocessing
from data_preprocessing import Vocab,Context,Dataset
import tempfile
from model import word_to_tokens
from model2 import bert_preprocess
from bandit import return_answer_index,generate_reward

np.set_printoptions(precision=4, suppress=True)


def process_trecqa(probs,rank,qid):
    rank = list(rank)
    res=[]
    for i,score in enumerate(probs):
        s="\t".join((str(qid),'Q0',str(i),str(rank.index(i)),str(score),"STANDARD"))
        s=s+"\n"
        res.append(s)
    return res


def reinforce_loss(probs, context,
                   max_num_of_ans=3):
    # sample sentences
    probs_numpy = probs.data.cpu().numpy()
    probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
    max_num_of_ans = min(len(probs_numpy), max_num_of_ans)  # max of sents# in doc and sents# in summary
    answer_index_list,_ = return_answer_index(probs_numpy,probs,sample_method="greedy",max_num_of_ans=len(probs_numpy))
    gold_index_list = context.labels
    #print(gold_index_list)
    #print(answer_index_list)
    reward = generate_reward(gold_index_list,answer_index_list)
    return reward,answer_index_list


def ext_model_eval(model, vocab, args, eval_data="test",trec_file="trecfile.txt"):
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
            print("Done %2d chunck, %4d/%4d context\r" % (phase+1, step + 1, len(dataset)),end=' ')
            
            context = contexts[0]

            # if args.oracle_length == -1:  # use true oracle length
            #     oracle_summary_sent_num = len(doc.summary)
            # else:
            #     oracle_summary_sent_num = args.oracle_length
            q,a,a_len,_ = word_to_tokens(context.question,context.answers,vocab)
            q = torch.autograd.Variable(q).cuda()
            a = torch.autograd.Variable(a).cuda()
            a_len.cuda()
            outputs = model(q,a,a_len)

            #pre_processed,a_len,sorted_id = bert_preprocess(context.answers)
            #q_a = torch.autograd.Variable(pre_processed.type(torch.float))
            #a_len = torch.autograd.Variable(a_len)
            #outputs = model(q_a,a_len)
            # outputs = model(q,a,a_len)
            #context.labels = np.array(context.labels)[sorted_id]



            # compute_score = (step == len(dataset) - 1)

            reward,rank_list = reinforce_loss(outputs,context,max_num_of_ans=len(a_len))
            #trec_result.extend(process_trecqa(outputs.squeeze().data.cpu().numpy(),rank_list,step_in_epoch))
            eval_rewards.append(reward)
            step_in_epoch+=1
    avg_eval_r = np.mean(eval_rewards, axis=0)
    print('model reward in %s:' % (eval_data))
    print(avg_eval_r)
    #if eval_data == "test":
    #    with open(trec_file,"w") as f:
    #        for row in trec_result:
    #            f.write(row)
    return avg_eval_r


if __name__ == '__main__':
    import pickle

    torch.manual_seed(1234)
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_file', type=str, default='trec_qa/pickle_data/vocab.42b.300d.p')
    parser.add_argument('--data_dir', type=str, default='trec_qa/pickle_data/')
    parser.add_argument('--model_file', type=str, default='model/select.ans')
    parser.add_argument('--trec_file',type=str,default="trec_file.txt")
    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--rl_baseline_method', type=str, default="greedy",
                        help='greedy, global_avg,batch_avg,or none')
    parser.add_argument('--length_limit', type=int, default=-1,
                        help='length limit output')

    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    print('generate config')
    with open(args.vocab_file, "rb") as f:
        vocab = pickle.load(f)
    print(vocab)

    print("loading existing model%s" % args.model_file)
    mcan_cb = torch.load(args.model_file, map_location=lambda storage, loc: storage)
    mcan_cb.cuda()
    print("finish loading and evaluate model %s" % args.model_file)
    data = args.data_dir.split('/')[0]
    start_time = time.time()
    if data == "insurance_qa":
        ext_model_eval(mcan_cb, vocab, args, eval_data="test1",trec_file= args.trec_file)
        ext_model_eval(mcan_cb, vocab, args, eval_data="test2",trec_file= args.trec_file)
    else :
        ext_model_eval(mcan_cb, vocab, args, eval_data="test",trec_file= args.trec_file)
    print('Test time:', time.time() - start_time)
