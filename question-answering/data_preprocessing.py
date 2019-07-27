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
    def __init__(self,question,answers,labels):
        assert(len(answers)==len(labels))
        self.question=question
        self.answers=answers
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

class Vocab():
    def __init__(self):
        self.word_list=['<pad>','<unk>']
        self.word_to_index={}
        self.index_to_word={}
        self.count=0
        self.embedding=None
    def __getitem__(self,key):
        if key in self.word_to_index.keys():
            return self.word_to_index[key]
        else:
            return self.word_to_index['<unk>']
    def add_vocab(self, vocab_file="data/vocab.txt"):
        with open(vocab_file, "rb") as f:
            for line in f:
                self.word_list.append(line.split()[0])  
        print("read %d words from vocab file" % len(self.word_list))
        for w in self.word_list:
            self.word_to_index[w] = self.count
            self.index_to_word[self.count] = w
            self.count += 1
    def add_embedding(self, gloveFile="glove/glove.6B.300d.txt", embed_size=300,init=False):
        print("Loading Glove embeddings")
        with open(gloveFile, 'rb') as f:
            model = {}
            w_set = set(self.word_list)
            if init:
                embedding_matrix = np.random.randn(len(self.word_list), embed_size)
            else:
                embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.word_to_index[word]] = embedding
                    if len(model) % 1000 == 0:
                        print("processed %d data" % len(model))
        embedding_matrix[1] = np.random.multivariate_normal(np.zeros(embed_size),np.eye(embed_size)) # for unkown words
        self.embedding = embedding_matrix
        print("%d words out of %d has embeddings in the glove file" % (len(model), len(self.word_list)))

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
        if data_quota == -1: #none-quota randomize data
            random.seed()
            random.shuffle(os_list)

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

def generate_vocab(vocab_file,outfile="trec_qa/pickle_data/vocab.300d.p",gloveFile="glove/glove.6B.300d.txt", embed_size=300,init=False):
    vocab=Vocab()
    vocab.add_vocab(vocab_file)
    vocab.add_embedding(gloveFile=gloveFile,embed_size=embed_size,init=init)
    pickle.dump(vocab,open(outfile,"wb+"))

def write_to_pickle(in_file, out_file, chunk_size=1000):
    boundaries = []
    with open(os.path.join(in_file,"boundary.txt")) as f:
        boundaries = f.readlines()
    boundaries=list(map(int,boundaries))
    labels =[]
    with open(os.path.join(in_file,"sim.txt")) as f:
        labels = f.readlines()
    labels=list(map(int,labels))
    i=0
    b=1
    contexts=[]
    answers_list=[]
    question_file= os.path.join(in_file,"a.toks")
    with open(os.path.join(in_file,"b.toks")) as answers:
        for _,a in tqdm(enumerate(answers),desc=in_file):
            answers_list.append(a.strip())
            i+=1
            if boundaries[b]==i :
                q=linecache.getline(question_file,i)
                contexts.append(Context(q.strip(),answers_list,labels[boundaries[b-1]:boundaries[b]]))
                answers_list=[]
                if b % chunk_size ==0 :
                    pickle.dump(Dataset(contexts),open(out_file%(b/chunk_size),"wb+"))
                    contexts=[]
                b+=1
    if contexts != []:
        pickle.dump(Dataset(contexts),open(out_file%(b/chunk_size+1),"wb+"))
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--vocabulary',action='store_true',help="Build Vocabulary.Requires\
                                                                    vocabulary file and the embedding files as input")
    parser.add_argument('-vi','--vocabulary_input',type=str,default="trec_qa/vocab.txt")
    parser.add_argument('-init','--init_emb', action ='store_true',default=False)
    parser.add_argument('-emb_size','--embedding_size',type=int,default=300)
    parser.add_argument('-vo','--vocabulary_output',type=str,default="trec_qa/pickle_data/vocab.300d.p")    
    parser.add_argument('-emb','--embedding_file',type=str,default="glove/glove.6B.300d.txt")
    parser.add_argument('-p','--parse_qa',action='store_true',help="Parse the Question and Answers into Context Objects For Training")
    parser.add_argument('-p_train','--parse_train',type=str,default="trec_qa/train-all")
    parser.add_argument('-p_val','--parse_val',type=str,default="trec_qa/clean-dev")
    parser.add_argument('-p_test','--parse_test',type=str,default="trec_qa/clean-test")
    parser.add_argument('-p_out','--parse_output',type=str,default="trec_qa/pickle_data/chunked")

    args=parser.parse_args()

    if args.vocabulary:
        vocab= args.vocabulary_input
        gloveFile=args.embedding_file

        generate_vocab(vocab,args.vocabulary_output,gloveFile,embed_size=args.embedding_size,init=args.init_emb)
    if args.parse_qa:
        train = args.parse_train
        val = args.parse_val
        test = args.parse_test
        data = train.split("/")[0]
        if data == "insurance_qa":
            write_to_pickle(test+"/test1", os.path.join(args.parse_output,"test1_%03d.bin.p"), chunk_size=100000000)
            write_to_pickle(test+"/test2", os.path.join(args.parse_output,"test2_%03d.bin.p"), chunk_size=100000000)
        else:
            write_to_pickle(test, os.path.join(args.parse_output,"test_%03d.bin.p"), chunk_size=100000000)
        write_to_pickle(val, os.path.join(args.parse_output,"val_%03d.bin.p"), chunk_size=100000000)
        write_to_pickle(train, os.path.join(args.parse_output,"train_%03d.bin.p"))

if __name__=='__main__':
    main()
