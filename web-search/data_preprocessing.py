import numpy as np 
import random
import argparse
import pickle
import os
import linecache
from tqdm import tqdm 

"""
Modified/Adapted from https://github.com/yuedongP/BanditSum

"""
class Context():
    def __init__(self,features,labels):
        assert(len(features)==len(labels))
        self.features=features
        self.labels=labels

class Dataset():
    def __init__(self,context_list):
        self._data=context_list
    def __len__(self):
        return len(self._data)
    def __call__(self,batch_size,shuffle=True):
        total_size=len(self)
        if shuffle:
            random.shuffle(self._data)
        batches = [self._data[i:i+batch_size] for i in range(0,total_size,batch_size)]
        return batches
    def __getitem__(self,index):
        return self._data[index]

class BatchDataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=True):
        assert isinstance(dataset, Dataset)
        assert len(dataset) >= batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.dataset(self.batch_size, self.shuffle))
class PickleReader():
    """

    this class intends to read pickle files converted by RawReader

    """

    def __init__(self, pickle_data_dir="trec_qa/pickle_data/"):
        """
        :param pickle_data_dir: the base_dir where the pickle data are stored in
        this dir should contain train.p, val.p, test.p, and vocab.p
        this dir should also contain the chunked_data folder
        """
        self.base_dir = pickle_data_dir

    def data_reader(self, dataset_path):
        """
        :param dataset_path: path for data.p
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        return data

    def full_data_reader(self, dataset_type="train"):
        """
        this method read the full dataset
        :param dataset_type: "train", "val", or "test"
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        return self.data_reader(self.base_dir + dataset_type + ".p")

    def chunked_data_reader(self, dataset_type="train", data_quota=-1):
        """
        this method reads the chunked data in the chunked_data folder
        :return: a iterator of chunks of datasets
        """
        data_counter = 0
        # chunked_dir = self.base_dir + "chunked/"
        chunked_dir = os.path.join(self.base_dir, 'chunked')
        os_list = os.listdir(chunked_dir)
        #if data_quota == -1: #none-quota randomize data
        #    random.seed()
        #    random.shuffle(os_list)

        for filename in os_list:
            if filename.startswith(dataset_type):
                # print("filename:", filename)
                chunk_data = self.data_reader(os.path.join(chunked_dir, filename))
                if data_quota != -1:  # cut off applied
                    quota_left = data_quota - data_counter
                    # print("quota_left", quota_left)
                    if quota_left <= 0:  # no more quota
                        break
                    elif quota_left > 0 and quota_left < len(chunk_data):  # return partial data
                        yield Dataset(chunk_data[:quota_left])
                        break
                    else:
                        data_counter += len(chunk_data)
                        yield chunk_data
                else:
                    yield chunk_data
            else:
                continue

def check_file(in_file,init_qid):
    unique=[]
    relevance_scores = set()
    prev = -1
    flag=0
    with open(in_file) as data:
        for line in data:
                qid = int(line.split(' ')[1].split(':')[1])
                relevance_scores.add(int(line.split(' ')[0]))
                if prev != qid :
                    prev =qid 
                    if flag ==0 :
                        assert(qid==init_qid)   
                        flag=1
                       
                    if qid in unique:
                        print("failure")
                        return 0,relevance_scores
                    else:
                        unique.append(qid)
    return 1,relevance_scores



def write_to_pickle(in_file, out_file,feat_start,emb_size,init_qid=-1,chunk_size=250):
    assert(init_qid !=-1)
    check = check_file(in_file,init_qid)
    assert(check[0] ==1)
    print("Relevance Levels:"+str(check[1]))
    i=0
    b=0
    maxi=-1
    contexts=[]
    labels=[]
    answers_list=[]
    prev = init_qid
    with open(os.path.join(in_file)) as answers:
        for line_num,a in tqdm(enumerate(answers),desc=in_file):
            line = a.split(' ')
            qid = int(line[1].split(':')[1])
            if prev != qid : 
                prev = qid                
                maxi = max(maxi,sum(np.asarray(labels)>0))
                contexts.append(Context(np.array(answers_list),np.array(labels)))
                b+=1
                if b % chunk_size ==0 :
                    pickle.dump(Dataset(contexts),open(out_file%(b/chunk_size),"wb+"))
                    contexts=[]
                answers_list=[]
                labels = []

            labels.append(int(line[0]))
            features = line[feat_start:feat_start+emb_size]
            answers_list.append(np.array([float(k.split(':')[1]) for k in features]))
    maxi = max(maxi,sum(np.asarray(labels)>0))
    contexts.append(Context(np.array(answers_list),np.array(labels)))     
    if contexts != []:
        pickle.dump(Dataset(contexts),open(out_file%(b/chunk_size+1),"wb+"))
    print("max number of relevant docs:%d"%maxi)
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-emb_size','--embedding_size',type=int,default=46)
    parser.add_argument('-train','--parse_train',type=str,default="MQ2007")
    parser.add_argument('-val','--parse_val',type=str,default="MQ2007")
    parser.add_argument('-test','--parse_test',type=str,default="MQ2007/Fold1/test.txt")
    parser.add_argument('-out','--parse_output',type=str,default="MQ2007/Fold1/chunked")
    parser.add_argument('-val_init','--val_init',type=int,default=-1)
    parser.add_argument('-test_init','--test_init',type=int,default=7968)
    parser.add_argument('-train_init','--train_init',type=int,default=-1)
    parser.add_argument('-feat_start','--feat_start',type=int,default=2)




    args=parser.parse_args()

    assert(args.test_init != -1)
   
    train = args.parse_train
    val = args.parse_val
    test = args.parse_test

    write_to_pickle(test, os.path.join(args.parse_output,"test_%03d.bin.p"),args.feat_start,args.embedding_size, init_qid=args.test_init,chunk_size=100000000)

    write_to_pickle(val, os.path.join(args.parse_output,"val_%03d.bin.p"), args.feat_start,args.embedding_size,init_qid=args.val_init,chunk_size=100000000)
    write_to_pickle(train, os.path.join(args.parse_output,"train_%03d.bin.p"),args.feat_start,args.embedding_size,init_qid=args.train_init)

if __name__=='__main__':
    main()
