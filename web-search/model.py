import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from torch.nn import Parameter
from copy import deepcopy
import numpy as np

class Highway(nn.Module):
    #https://github.com/kefirski/pytorch_Highway/blob/master/highway/highway.py
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        #self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            #linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * x

        return x



class MCAN_CB(nn.Module):
    def __init__(self,config):
        super(MCAN_CB,self).__init__()
        
        # Parameters
        self.drop = config.dropout

        self.input_dim = config.input_dim
        #self.mid = int(self.input_dim/2)
        self.mid = config.mid_dim
        #self.highway = config.highway
        #self.multi_class = config.multi_class
        #self.num_class = config.num_class
        #self.nn_layers = config.nn_layers
        #if self.multi_class:
        #    self.multi_class = nn.Sequential(nn.Linear(self.input_dim,self.num_class),                                              nn.Softmax() )

        # if config.highway:
        #if self.highway:
        #    self.decoder = nn.Sequential(nn.Linear(self.input_dim,self.input_dim)
        #                                ,nn.ReLU(),
        #                                nn.Dropout(self.drop),
        #                                Highway(self.input_dim,2,F.relu),
        #                                nn.Dropout(self.drop))
        #    self.bandit = nn.Sequential(nn.Linear(self.input_dim,1),
        #                              nn.Sigmoid())
        if config.highway:
            self.decoder = nn.Sequential(nn.Linear(self.input_dim,self.mid),
                                        nn.ReLU(),
                                        #nn.Tanh(),
                                        nn.Dropout(self.drop),
                                        Highway(self.mid,3,F.relu),
                                        nn.Dropout(self.drop),
                                      nn.Linear(self.mid,1),
                                      nn.Sigmoid())        
        #if self.multi_class:
        #    self.forward = self.forward_2
        #else:
        #    self.forward = self.forward_1
       # elif self.nn_layers == 1 :
         #   self.decoder = nn.Sequential(nn.Linear(self.input_dim,self.input_dim),
         #                            nn.Tanh(),
         #                            nn.Dropout(self.drop),
         #                            nn.Linear(self.input_dim,1),
         #                            nn.Sigmoid())
        #elif self.nn_layers == 2:
         #   self.decoder = nn.Sequential(nn.Linear(self.input_dim,self.input_dim),
         #                            nn.Tanh(),
         #                            nn.Dropout(self.drop),
         #                            nn.Linear(self.input_dim,int(self.input_dim/2)),
         #                            nn.Tanh(),
         #                            nn.Dropout(self.drop),
         #                            nn.Linear(int(self.input_dim/2),1),
         #                            nn.Sigmoid())


    #def forward_1(self,q_d): #input context tokens      
    #    prob_inp = self.decoder(q_d)
    #    prob = self.bandit(prob_inp)
    #    return [prob[:,0]]
    #def forward_2(self,q_d):
    #    prob_inp = self.decoder(q_d)
    #    prob1 = self.bandit(prob_inp)
    #    prob2 = self.multi_class(prob_inp)
    #    return [prob1[:,0],prob2]
    def forward(self,q_d): #input context tokens      
        prob = self.decoder(q_d)
        return prob[:,0]




        
