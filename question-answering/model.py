import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from torch.nn import Parameter
from copy import deepcopy
import numpy as np

class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

       # self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

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

class Compression_FM(nn.Module):
    #Reference:  https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch
    #original code was for single sequence, modified it to batch of sequence

    def __init__(self, n=None, k=None):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k),requires_grad=True)
        self.lin = nn.Linear(n, 1)

        
    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(2, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(2, keepdim=True) # S_2
        
        #https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf (lemma 3.1)
        out_inter = 0.5*(out_1 - out_2)        
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out

def compression_SM(x,dim=2):
    return x.sum(dim=dim)

def word_to_tokens(q,a,vocab):
    q_temp = q.lower().strip().split()
    q_tok = torch.LongTensor([[vocab[word.encode('utf-8')] for word in q_temp]])
    a_tok =[]
    max_len=-1
    a_len=[]
    for sent in a:
        words = sent.lower().strip().split()
        sent = [vocab[word.encode('utf-8')] for word in words]
        if len(sent)==0:
            continue        
        a_len.append(len(sent))
        a_tok.append(torch.Tensor(sent)) 

    a_tok = pad_sequence(a_tok,batch_first=True,padding_value=0).long()  # pad with 0's i.e <pad> tokens
    max_len = a_tok.size()[-1]
    return q_tok,a_tok,torch.LongTensor(a_len),max_len

class MCAN_CB(nn.Module):
    def __init__(self,config):
        super(MCAN_CB,self).__init__()
        
        # Parameters
        self.drop = config.dropout
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.word_input_size = config.word_input_size  #embedding size +12 (4*3)
        self.LSTM_hidden_units = config.LSTM_hidden_units
        self.compression_type = config.compression_type #SM,NN,FM
        self.num_factors = config.num_factors # k in FM



        #Layers
        self.dropout = nn.Dropout(self.drop)
        self.word_embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.word_embedding.weight.requires_grad=False

        self.co_attention = nn.Sequential(nn.Linear(self.embedding_dim,self.embedding_dim),
                                          nn.ReLU())
        self.intra_attention_q = nn.Sequential(nn.Linear(self.embedding_dim,self.embedding_dim),
                                    nn.ReLU())
        self.intra_attention_a = nn.Sequential(nn.Linear(self.embedding_dim,self.embedding_dim),
                                          nn.ReLU())                         
        if self.compression_type == "NN":
            self.cast_attention_product = nn.Sequential(nn.Linear(self.embedding_dim,1),
                                            nn.ReLU())
            self.cast_attention_diff  = nn.Sequential(nn.Linear(self.embedding_dim,1),
                                            nn.ReLU())
            self.cast_attention_concat = nn.Sequential(nn.Linear(2*self.embedding_dim,1),
                                            nn.ReLU())
        elif self.compression_type == "FM":
            self.cast_attention_product = Compression_FM(n=self.embedding_dim,k=self.num_factors)
            self.cast_attention_diff = Compression_FM(n=self.embedding_dim,k=self.num_factors)
            self.cast_attention_concat = Compression_FM(n= 2*self.embedding_dim,k=self.num_factors)
        else :
            self.cast_attention_product = compression_SM
            self.cast_attention_diff = compression_SM
            self.cast_attention_concat = compression_SM
       # if self.compression_type == "NN":
       #     self.cast_attention_diff = [nn.Sequential(nn.Linear(self.embedding_dim,1),
        #                                    nn.ReLU()) for i in range(4) ]
       #     self.cast_attention_product = [nn.Sequential(nn.Linear(self.embedding_dim,1),
       #                                     nn.ReLU()) for i in range(4) ]
       #     self.cast_attention_concat = [nn.Sequential(nn.Linear(2*self.embedding_dim,1),
       #                                     nn.ReLU()) for i in  range(4)]
       # elif self.compression_type == "FM":
       #     self.cast_attention_diff = nn.ModuleList([Compression_FM(n=self.embedding_dim,k=self.num_factors) for i in  range(4) ])
       #     self.cast_attention_product = nn.ModuleList([Compression_FM(n=self.embedding_dim,k=self.num_factors) for i in  range(4) ]) 
       #     self.cast_attention_concat = nn.ModuleList([Compression_FM(n= 2*self.embedding_dim,k=self.num_factors) for i in  range(4) ]) 
       # else :
       #     self.cast_attention_diff = [compression_SM for i in  range(4) ] 
       #     self.cast_attention_product = [compression_SM for i in  range(4) ]
       #     self.cast_attention_concat = [compression_SM for i in  range(4) ] 

        self.word_LSTM = nn.LSTM(
                input_size = self.word_input_size,
                hidden_size = self.LSTM_hidden_units,
                batch_first = True,
                dropout=self.drop                
        )
        #print(config.highway)
        #print(80*'*')
        if config.highway:
            self.decoder = nn.Sequential( nn.Linear(8*self.LSTM_hidden_units,200),
                                          nn.ReLU(),
                                          nn.Dropout(self.drop),
                                        Highway(200,2,F.relu),
                                      nn.Dropout(self.drop),
                                      nn.Linear(200,2),
                                      nn.Softmax())
        else:
            self.decoder = nn.Sequential(nn.Linear(8*self.LSTM_hidden_units,200),
                                     nn.Tanh(),
                                     nn.Linear(200,1),
                                     nn.Softmax())
        #self.decoder = nn.Sequential(nn.Linear(2*self.LSTM_hidden_units,100),
         #                            nn.Tanh(),
          #                           nn.Linear(100,1),
           #                          nn.Sigmoid())
       # self.decoder = nn.Sequential(Highway(2*self.LSTM_hidden_units,2,F.relu),
        #                              nn.Linear(2*self.LSTM_hidden_units,1),
         #                             nn.Sigmoid())


    def multi_cast_attention(self,q,a,a_len,batch_size): # shape (batch_size,q_l,emb_dim) (batch_size,a_l,emb_dim)
        #batch_size here refers to number of candidate answers for a given question
        q_shape= q.size()  # (1,q_l,emb)
        q_att = self.co_attention(q.view(-1,self.embedding_dim)) # (1*q_l,emb)
        q_att = self.dropout(q_att)
        q_att= q_att.view(q_shape) #back to normal shape
        
        a_shape= a.size() #(batch,a_l,emb)
        #(batch,a_l)
        a_mask = torch.arange(a_shape[1])[None, :] < a_len[:, None]  #for removing influence of <pad>
        a_mask = a_mask.view(-1,a_shape[1],1).cuda()
        a  = a.masked_fill(a_mask==0,0.)
        a_att = self.co_attention(a.view(-1,self.embedding_dim)) # (batch*a_l,emb)
        a_att = self.dropout(a_att)
        a_att= a_att.view(a_shape) #back to normal

        a_att  = a_att.masked_fill(a_mask==0,0.)  #zeroing the padding embeddings
        a_att_t= a_att.permute(0,2,1) #(batch,emb,a_l)    

        s = q_att.matmul(a_att_t) #(batch,q_l,a_l)  Affinity Matrix s , broadcast q_att
        s_inf =s.transpose(1,2).masked_fill(a_mask==0,-1e9).transpose(1,2).cuda()
        s_zero =s.transpose(1,2).masked_fill(a_mask==0,0.).transpose(1,2).cuda()

        #s_zero =s.clone()
        #s_zero= s_zero.transpose(1,2)  # (batch,a_l,q_l)
        #s_zero[~a_mask,:]= torch.zeros(q_shape[1]).cuda()
        #s_zero = s_zero.transpose(1,2)


        #for handling softmax and max of <pad> during attention
        # refer http://juditacs.github.io/2018/12/27/masked-attention.html
        #s_inf =s.clone()
        #s_inf= s_inf.transpose(1,2)  # (batch,a_l,q_l)
        #s_inf[~a_mask,:]= torch.Tensor(q_shape[1]).fill_(float('-inf')).cuda()
        #s_inf = s_inf.transpose(1,2)

        # Calculating all the attentions
        # 1) EXTRACTIVE MAX and MEAN pooling

        s_max_1 = torch.max(s_inf,dim=1).values.view(batch_size,1,-1)  # (batch,1,a_l)
        s_max_2 = torch.max(s_inf,dim=2).values.view(batch_size,1,-1)  # (batch,1,q_l)
        s_mean_1 = torch.mean(s_zero,dim=1).view(batch_size,1,-1)  # (batch,1,a_l)
        s_mean_2 = (torch.sum(s_zero,dim=2)/a_len.view(-1,1).type(torch.float).cuda()).view(batch_size,1,-1)  # (batch,1,q_l)
        #s_mean_2 = torch.mean(s,dim=2).view(batch_size,1,-1)

        q_max = F.softmax(s_max_2,dim=2).matmul(q) # (batch,1,emb)
        q_mean = F.softmax(s_mean_2,dim=2).matmul(q) # (batch,1,emb)

        a_max = torch.bmm(F.softmax(s_max_1,dim=2),a) # (batch,1,emb)
        a_mean = torch.bmm(F.softmax(s_mean_1,dim=2),a) # (batch,1,emb)

        # 2) ALIGNMENT pooling
        s_2 =s.transpose(1,2).masked_fill(a_mask==0,-1e9).transpose(1,2).cuda()
        A_q = F.softmax(s_2,dim=2)  # sum of row =1 (weights of answer words)
        #A_q = self.dropout(A_q)
        q_align = torch.bmm(A_q,a) # question in terms of answer words (batch,q_l,emb)
        #s_inf_1 = s_inf.clone()
        A_a = F.softmax(s_2,dim=1) # sum of col =1 (weights of question words)
        A_a = A_a.transpose(1,2)
        #A_a = self.dropout(A_a)
        a_align = torch.matmul(A_a,q)  # answer in terms of question words  (batch,a_l,emb)

        # 3) INTRA Attention

        q_intra_att = self.intra_attention_q(q.view(-1,self.embedding_dim)).view(q_shape)
        q_intra_att = self.dropout(q_intra_att)
        s_q = q_intra_att.matmul(q_intra_att.transpose(1,2)) 
        q_intra = F.softmax(s_q,dim=2).matmul(q) #(1,q_l,emb)

        a_intra_att = self.intra_attention_a(a.view(-1,self.embedding_dim)).view(a_shape)
        a_intra_att = self.dropout(a_intra_att)
        s_a = a_intra_att.matmul(a_intra_att.transpose(1,2)).masked_fill(a_mask==0,-1e9) 
        #s_a_inf = s_a.clone()
        #s_a_inf[~a_mask,:] = torch.Tensor(a_shape[1]).fill_(float('-inf')).cuda()

        a_intra = F.softmax(s_a,dim=2).matmul(a)  #(batch,a_l,emb)  
        #a_intra = self.dropout(a_intra)

        # Casting the Attentions using compression function
        q = q.expand(batch_size,q_shape[1],self.embedding_dim)
        q_max = q_max.expand(batch_size,q_shape[1],self.embedding_dim)
        q_mean = q_mean.expand(batch_size,q_shape[1],self.embedding_dim)
        q_intra = q_intra.expand(batch_size,q_shape[1],self.embedding_dim)

        a_max = a_max.expand(batch_size,a_shape[1],self.embedding_dim)
        a_mean = a_mean.expand(batch_size,a_shape[1],self.embedding_dim)
        
        q_attention = [q_max,q_mean,q_align,q_intra]
        a_attention = [a_max,a_mean,a_align,a_intra]
        q_cast = []
        a_cast =[]
        # concat,product,subtract
        for i in range(0,4):
            q_cast.append(self.cast_attention_concat(torch.cat((q_attention[i],q),-1)))
            q_cast.append(self.cast_attention_product(q_attention[i]*q))
            q_cast.append(self.cast_attention_diff(q_attention[i]-q))
            a_cast.append(self.cast_attention_concat(torch.cat((a_attention[i],a),-1)))
            a_cast.append(self.cast_attention_product(a_attention[i]*a))
            a_cast.append(self.cast_attention_diff(a_attention[i]-a))
        for x,y in zip(q_cast,a_cast):
            q=torch.cat((q,x.view(-1,q_shape[1],1)),-1)
            a=torch.cat((a,y.view(-1,a_shape[1],1)),-1)

        return q,a

    def mean_max_pooling_a(self,x,a_len):
        a_mask = torch.arange(x.shape[1])[None, :] < a_len[:, None]  #for removing influence of <pad>
        x_zero = x.clone()
        x_zero[~a_mask,:] = torch.zeros(self.LSTM_hidden_units).cuda()
        h_mean = torch.sum(x_zero,dim=1)/a_len.view(-1,1).type(torch.float).cuda() #(a_l,h)
        x_inf = x.clone()
        x_inf[~a_mask,:] = torch.Tensor(self.LSTM_hidden_units).fill_(float('-inf')).cuda()
        h_max = torch.max(x_inf,dim=1).values  #(a_l,h)
        return torch.cat((h_mean,h_max),-1)


    def mean_max_pooling_q(self,x):
        h_mean= torch.mean(x,dim=1) #(num_ans,h)
        h_max = torch.max(x,dim=1).values  #(num_ans,h)
        return torch.cat((h_mean,h_max),-1)

    #def    mean_max_pooling(self,x,a_len):
    #     result=[]
    #     for index,data in enumerate(x):
    #         h_mean=torch.mean(data[:a_len[index],:],dim=0)
    #         h_max=torch.max(data[:a_len[index],:],dim=0).values
    #         result.append(torch.cat((h_mean,h_max),-1))
    #     return torch.cat(result,dim=0)


    def forward(self,q,a,a_len): #input context tokens      
        q = self.word_embedding(q)
        a = self.word_embedding(a)
        q,a = self.multi_cast_attention(q,a,a_len,len(a))

        #lstm_input = torch.cat((q,a),dim = 1) # (num_ans,q_l+max_len,emb+12)
        q_out,hidden = self.word_LSTM(q)
        a_out,_ = self.word_LSTM(a,hidden)
        h_mean_max_q = self.mean_max_pooling_q(q_out)
        #h_mean_max_q =self.dropout(h_mean_max_q)
        h_mean_max_a = self.mean_max_pooling_a(a_out,a_len)
        #h_mean_max_a = self.dropout(h_mean_max_a)  #(num_ans,2*h)
        decoder_inp = torch.cat([h_mean_max_q,h_mean_max_a,
                                h_mean_max_a*h_mean_max_q,
                                h_mean_max_q-h_mean_max_a],dim=-1)

        prob = self.decoder(decoder_inp.view(-1,self.LSTM_hidden_units*8))

        return prob[:,0].view(-1,1)





        
