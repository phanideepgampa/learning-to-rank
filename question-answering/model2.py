import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
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

def bert_preprocess(answers):
    seq_lengths = list(map(len,answers))
    input_dim = len(answers[0][0])
    #print(seq_lengths)
    answers = np.array(answers)
    #print(len(answers))
    padded_array = np.zeros((answers.shape[0],max(seq_lengths),input_dim))
    for idx, (seq, seqlen) in enumerate(zip(answers, seq_lengths)):
        padded_array[idx, :seqlen,:] = seq

    seq_lengths = torch.LongTensor(seq_lengths).cuda()
    padded_array = torch.from_numpy(padded_array).cuda()
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    padded_array = padded_array[perm_idx]
    return padded_array,seq_lengths,perm_idx.cpu().tolist()



class BERT_CB(nn.Module):
    def __init__(self,config):
        super(BERT_CB,self).__init__()
        
        # Parameters
        self.drop = config.dropout
        self.dropout = nn.Dropout(self.drop)
        self.input_dim = config.input_dim
        if self.input_dim == 1024:
            self.lstm_hidden=512
            self.decoder_input = 1024
        elif self.input_dim == 4*768:
            self.lstm_hidden = 768
            self.decoder_input = 2*768
        elif self.input_dim == 4*1024:
            self.lstm_hidden = 1024
            self.decoder_input = 2*1024 

        self.word_LSTM = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.lstm_hidden,
                num_layers=2,
                dropout=self.drop,
                batch_first=True,
                bidirectional=True)

        if config.highway:
            self.decoder = nn.Sequential(nn.Linear(self.decoder_input,256)
                                        ,nn.ReLU(),
                                        nn.Dropout(self.drop),
                                        Highway(256,2,F.relu),
                                        nn.Dropout(self.drop),
                                      nn.Linear(256,1),
                                      nn.Sigmoid())
        elif config.nn_layers == 1:
            self.decoder = nn.Sequential(nn.Linear(self.decoder_input,256),
                                      nn.Tanh(),
                                      nn.Dropout(self.drop),
                                      nn.Linear(256,1),
                                      nn.Sigmoid())
        elif config.nn_layers == 2:
            self.decoder = nn.Sequential(nn.Linear(self.decoder_input,512),
                                      nn.Tanh(),
                                      nn.Dropout(self.drop),
                                      nn.Linear(512,256),
                                      nn.Tanh(),
                                      nn.Dropout(self.drop),
                                      nn.Linear(256,1),
                                      nn.Sigmoid())



    def forward(self,q_d,a_len): #input context tokens      
        packed_input = pack_padded_sequence(q_d,a_len,batch_first =True)
        packed_output,_ = self.word_LSTM(packed_input)
        output,_ = pad_packed_sequence(packed_output,batch_first =True)
        mean_pool = torch.sum(output,dim=1)/a_len.view(-1,1).type(torch.float)
        mean_pool = self.dropout(mean_pool)
        prob = self.decoder(mean_pool)
        return prob[:,0]









# def word_to_tokens(q,a,vocab):
#     q_temp = q.lower().strip().split()
#     q_tok = torch.LongTensor([[vocab[word.encode('utf-8')] for word in q_temp]])
#     a_tok =[]
#     max_len=-1
#     a_len=[]
#     for sent in a:
#         words = sent.lower().strip().split()
#         sent = [vocab[word.encode('utf-8')] for word in words]
#         if len(sent)==0:
#             continue        
#         a_len.append(len(sent))
#         a_tok.append(torch.Tensor(sent)) 

#     a_tok = pad_sequence(a_tok,batch_first=True,padding_value=0).long()  # pad with 0's i.e <pad> tokens
#     max_len = a_tok.size()[-1]
#     return q_tok,a_tok,torch.LongTensor(a_len),max_len

# class CA_CB(nn.Module):
#     def __init__(self,config):
#         super(CA_CB,self).__init__()
        
#         # Parameters
#         self.drop = config.dropout
#         self.vocab_size = config.vocab_size
#         self.embedding_dim = config.embedding_dim
#         self.kernels = [1,2,3,4,5]  
#         self.hidden_units = int(self.embedding_dim/2)
#         self.Ci=1

#         self.convs1 = nn.ModuleList([nn.Conv2d(self.Ci, self.hidden_units, (K, self.hidden_units)) for K in self.kernels])

#         #Layers
#         self.dropout = nn.Dropout(self.drop)
#         self.word_embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
#         self.word_embedding.weight.requires_grad=False
#         self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))

#         self.preprocess1  = nn.Sequential(nn.Linear(self.embedding_dim,self.hidden_units),
#                                            nn.Sigmoid() )
#         self.preprocess2  = nn.Sequential(nn.Linear(self.embedding_dim,self.hidden_units),
#                                            nn.Tanh() )                                   
#         self.co_attention = nn.Sequential(nn.Linear(self.hidden_units,self.hidden_units),
#                                           nn.ReLU())                        
        
#         self.decoder = nn.Sequential(nn.Linear(self.hidden_units*len(self.kernels),200)
#                                     ,nn.ReLU(),Highway(200,2,F.relu),
#                                     nn.Linear(200,1),
#                                     nn.Sigmoid())




#     def preprocess_attention(self,q,a,a_len,batch_size): # shape (batch_size,q_l,emb_dim) (batch_size,a_l,emb_dim)
#         #batch_size here refers to number of candidate answers for a given question
#         q_shape= q.size()  # (1,q_l,emb)


#         q_pre = self.preprocess1(q.view(-1,self.embedding_dim))*self.preprocess2(q.view(-1,self.embedding_dim))
#         q_att = self.co_attention(q_pre.view(-1,self.hidden_units)) # (1*q_l,emb)
#         q_att = self.dropout(q_att)
#         q_att= q_att.view(1,-1,self.hidden_units) #back to normal shape
        
#         a_shape= a.size() #(batch,a_l,emb)
#         #(batch,a_l)
#         a_mask = torch.arange(a_shape[1])[None, :] < a_len[:, None]  #  (batch,a_l)  for removing influence of <pad>
#         a = a.masked_fill(a_mask.view(-1,a_shape[1],1)==0,0.)
#         a_pre = self.preprocess1(a.view(-1,self.embedding_dim))*self.preprocess2(a.view(-1,self.embedding_dim))
#         a_att_t= a_pre.permute(0,2,1) #(batch,hidd,a_l)    
#         s = q_att.matmul(a_att_t) #(batch,q_l,a_l)  Affinity Matrix s , broadcast q_att

#         #for handling softmax and max of <pad> during attention
#         # refer http://juditacs.github.io/2018/12/27/masked-attention.html
#         #  ALIGNMENT pooling

#         s_2 =s.transpose(1,2).masked_fill(a_mask.view(-1,a_shape[1],1)==0,-1e9).transpose(1,2)
#         A_a = F.softmax(s_2,dim=1) # sum of col =1 (weights of question words)
#         A_a = A_a.transpose(1,2)
#         a_align = torch.matmul(A_a,q_pre)  # answer in terms of question words  (batch,a_l,hidd)
#         return a_pre,a_align
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x

#     def forward(self,q,a,a_len): #input context tokens      
#         q = self.word_embedding(q)
#         a = self.word_embedding(a)
#         a_pre,a_align = self.preprocess_attention(q,a,a_len,len(a))
#         compare = a_pre * a_align #(batch,a_l,hidd)
#         compare = compare.unsqueeze(1) #(batch,1,a_l,hidd)
#         compare = self.dropout(compare)
#         conv_out = [ self.conv_and_pool(compare,conv) for conv in self.convs1  ]

#         decoder_inp = torch.cat(conv_out,1)
#         decoder_inp =self.dropout(decoder_inp)

#         prob = self.decoder(decoder_inp.view(-1,self.hidden_units*len(self.kernels)))

#         return prob[:,0]





        
