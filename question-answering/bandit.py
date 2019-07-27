import random

import numpy as np
import torch
from torch.autograd import Variable
from copy import deepcopy

from rank_metrics import mean_reciprocal_rank,average_precision,ndcg_at_k,dcg_at_k
# code modified/adapted from https://github.com/yuedongP/BanditSum




def return_answer_index(probs_numpy, probs_torch, sample_method="sample", max_num_of_ans=10,method ="herke"):
    """
    :param probs: numpy array of the probablities for all answers for a question
    :param sample_method: greedy or sample
    :param max_num_of_ans: max num of answers to be selected
    :return: a list of index for the selected ans
    """
    assert isinstance(sample_method, str)
    if max_num_of_ans <= 0:
        if sample_method == "sample":
            l = np.random.binomial(1, probs_numpy)
        elif sample_method == "greedy":
            l = [1 if prob >= 0.5 else 0 for prob in probs_numpy]
        answer_index = np.nonzero(l)[0]
    else:
        if sample_method == "sample":
            probs_torch = probs_torch.squeeze()
            if len(probs_torch.size()) != 1:
                print(probs_torch)
                print(probs_torch.size()==torch.Size([]))

            if method == 'original':
                # original method
                probs_clip = probs_numpy * 0.8 + 0.1
                # print("sampling the index for the answer")
                index = range(len(probs_clip))
                probs_clip_norm = probs_clip / sum(probs_clip)
                answer_index = np.random.choice(index, max_num_of_ans, replace=False,
                                                 p=np.reshape(probs_clip_norm, len(probs_clip_norm)))
                p_answer_index = probs_numpy[answer_index]
                sorted_idx = np.argsort(p_answer_index)[::-1]
                answer_index = answer_index[sorted_idx]
                loss = 0.
                for idx in index:
                    if idx in answer_index:
                        loss += probs_torch[idx].log()
                    else:
                        loss += (1 - probs_torch[idx]).log()
            elif method == 'herke':
                # herke's method
                answer_index = []
                epsilon = 0.1
                mask = Variable(torch.ones(probs_torch.size()).cuda(), requires_grad=False)
                # mask = Variable(torch.ones(probs_torch.size()), requires_grad=False)
                loss_list = []
                for i in range(max_num_of_ans):
                    p_masked = probs_torch * mask
                    if random.uniform(0, 1) <= epsilon:  # explore
                        selected_idx = torch.multinomial(mask, 1)
                    else:
                        selected_idx = torch.multinomial(p_masked, 1)
                    loss_i = (epsilon / mask.sum() + (1 - epsilon) * p_masked[selected_idx] / p_masked.sum()).log()
                    loss_list.append(loss_i)
                    mask = mask.clone()
                    mask[selected_idx] = 0
                    answer_index.append(selected_idx)

                answer_index = torch.cat(answer_index, dim=0)
                answer_index = answer_index.data.cpu().numpy()

                loss = sum(loss_list)
        elif sample_method == "greedy":
            loss = 0
            answer_index = np.argsort(np.reshape(probs_numpy, len(probs_numpy)))[-max_num_of_ans:]
            answer_index = answer_index[::-1]

    # answer_index.sort()
    return answer_index, loss

"""
stats of TrecQa: Number of Answers for a Question
max ans:761
min ans:1
mean ans:43.4

    #calculate 
    for rank,ans in enumerate(answer_index_list):
        if gold_index_list[ans]==1:
            reciprocal_rank=1/(rank+1)
            break
    num_relevant =0
    for rank,ans in enumerate(answer_index_list):
        if gold_index_list[ans]==1:
            num_relevant+=1
            average_precision+=(num_relevant/(rank+1))
    average_precision/= sum(gold_index_list)
"""


def generate_reward(gold_index_list, answer_index_list,reward_type=1):
    reward = 0
    ap=0
    reciprocal_rank=0
    answer_list =list(deepcopy(answer_index_list))
    size =len(answer_index_list)
    true = sum(gold_index_list)
    inp = np.zeros(size)
    for rank,val in enumerate(gold_index_list) :
        if val and rank in answer_list:
            inp[answer_list.index(rank)]=2
    if true:
        ap=average_precision(inp)*(sum(inp>0)/true)
    reciprocal_rank = mean_reciprocal_rank([inp])
    #ndcg = ndcg_at_k(inp,size)
    #if reward_type==1:
    #    reward = (ap+reciprocal_rank)/2
    #elif reward_type ==2 :
    #    reward = dcg_at_k(inp,size)
    rewards = [(ap+reciprocal_rank)/2,dcg_at_k(inp,size)]
    return rewards[reward_type-1],ap,reciprocal_rank,(inp[0]>0)



class ContextualBandit():
    def __init__(self,b=20, rl_baseline_method="greedy",sample_method="herke"):
        self.probs_torch = None
        self.probs_numpy = None
        self.max_num_of_ans = None
        self.context = None
        self.method = sample_method
        self.global_avg_reward = 0.
        self.train_examples_seen = 0.
        
        self.rl_baseline_method = rl_baseline_method
        self.b = b  # batch_size for calculating the gradient

    def train(self,prob,context,max_num_of_ans=10,reward_type=1,prt=False):
        """
        :return: training_loss_of_the current example
        """
        self.update_data_instance(prob, context, max_num_of_ans)
        self.train_examples_seen += 1
        gold_index_list = self.context.labels
        if len(self.probs_numpy)==1:
            #print(self.probs_torch)
            #print(80*'*')
            result = self.probs_torch.item() > 0.5
            reward = (result == gold_index_list[0])
            #print(result)
            if reward:
                loss = self.probs_torch.log()
            else:
                loss = (1-self.probs_torch).log()
            return loss.view(-1),reward
        batch_index_and_loss_lists = self.sample_batch(self.b)        
        batch_rewards = [
            generate_reward(gold_index_list,idx_list[0],reward_type)[0]
            for idx_list in batch_index_and_loss_lists
        ]

        rl_baseline_reward = self.compute_baseline(batch_rewards)
        loss = self.generate_batch_loss(batch_index_and_loss_lists, batch_rewards, rl_baseline_reward)
        #if loss == 0 :
        #    loss = torch.tensor(0.0)
        greedy_index_list, _ = self.generate_index_list_and_loss("greedy")
        greedy_reward,_,_,_ = generate_reward(gold_index_list,greedy_index_list,reward_type)

        if prt:
            print('Batch rewards:', np.array(batch_rewards))
            print('Greedy rewards:', np.array(greedy_reward))
            print('Baseline rewards:', np.array(rl_baseline_reward))

        return loss, greedy_reward

    def validate(self, probs, context, max_num_of_ans=3):
        """
        :return: validation_loss_of_the current example
        """
        self.update_data_instance(probs, context, max_num_of_ans)
        answer_index_list, _ = self.generate_index_list_and_loss("greedy")
        reward_tuple = generate_reward(self.context.labels,answer_index_list)
        return reward_tuple

    def update_data_instance(self, probs, context, max_num_of_ans=3):
        # self.probs_torch = probs
        # self.probs_torch = torch.clamp(probs, 1e-6, 1 - 1e-6)  # this just make sure no zero
        self.probs_torch = probs.clone() * 0.9999 + 0.00005  # this just make sure no zero
        probs_numpy = probs.data.cpu().numpy()
        self.probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
        self.context = context
        self.max_num_of_ans = min(len(probs_numpy), max_num_of_ans)

    def generate_index_list_and_loss(self, sample_method="sample"):
        """
        :param sample_method: "leadk,sample, greedy"
        :return: return a list of answer indexes for next step of computation
        """
        if sample_method == "lead_oracle":
            return range(self.max_num_of_ans), 0
        else:  # either "sample" or "greedy" based on the prob_list
            return return_answer_index(self.probs_numpy, self.probs_torch,
                                        sample_method=sample_method, max_num_of_ans=self.max_num_of_ans,method=self.method)

    def generate_answer(self, answer_index_list):
        return [self.context.answers[i] for i in answer_index_list]

    def sample_batch(self, b):
        batch_index_and_loss_lists = [self.generate_index_list_and_loss() for i in range(b)]
        return batch_index_and_loss_lists

    def compute_baseline(self, batch_rewards):
        def running_avg(t, old_mean, new_score):
            return (t - 1) / t * old_mean + new_score / t

        batch_avg_reward = np.mean(batch_rewards)
        batch_median_reward = np.median(batch_rewards)
        self.global_avg_reward = running_avg(self.train_examples_seen, self.global_avg_reward, batch_avg_reward)

        if self.rl_baseline_method == "batch_avg":
            return batch_avg_reward
        if self.rl_baseline_method == "batch_med":
            return batch_median_reward
        elif self.rl_baseline_method == "global_avg":
            return self.global_avg_reward
        elif self.rl_baseline_method == "greedy":
            answer_index_list, _ = self.generate_index_list_and_loss("greedy")
            return generate_reward(self.context.labels,answer_index_list)[0]
        else:  # none
            return 0

    def generate_batch_loss(self, batch_index_and_loss_lists, batch_rewards, rl_baseline_reward):
        loss_list = [
            batch_index_and_loss_lists[i][1] * ((rl_baseline_reward - batch_rewards[i]) / (rl_baseline_reward + 1e-9))
            for i in range(len(batch_rewards))
        ]
        avg_loss = sum(loss_list) / (float(len(loss_list)) + 1e-8)
        return avg_loss

 


