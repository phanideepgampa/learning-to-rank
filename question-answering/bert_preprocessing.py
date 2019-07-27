import json
import os
import pickle
import numpy as np
from tqdm import tqdm

from data_preprocessing import Context,Dataset

files = ["train","test","dev"]
directory = "wiki_qa/bert_large_pickle"
data ="wiki_qa"
chunk_size=100

for f in files:
    read = open(os.path.join(data,"bert_large_"+f+"_feature"))
    boundaries = []
    name = f
    if f == "dev":
        name="val"
    out_file = os.path.join(directory,"chunked",name+"_%03d.bin.p")
    with open(os.path.join(data,f,"boundary.txt")) as fi:
        boundaries = fi.readlines()
    boundaries=list(map(int,boundaries))
    labels =[]
    with open(os.path.join(data,f,"sim.txt")) as fi:
        labels = fi.readlines()
    labels=list(map(int,labels))
    i=0
    b=1
    contexts=[]
    answers_list=[]
    for line in tqdm(read,desc=f):
        answer =[]
        json_data = json.loads(line)
        for feature in json_data["features"]:
            concat =[]
            for layer in feature["layers"]:
                concat.extend(layer["values"])
            answer.append(np.array(concat)) 
            assert(len(concat)== 4*1024 )

        answers_list.append(np.array(answer))
        i+=1
        if boundaries[b]==i :
            contexts.append(Context(None,answers_list,labels[boundaries[b-1]:boundaries[b]]))
            answers_list=[]
            if b % chunk_size ==0 :
                pickle.dump(Dataset(contexts),open(out_file%(b/chunk_size),"wb+"))
                contexts=[]
            b+=1
    if contexts != []:
        pickle.dump(Dataset(contexts),open(out_file%(b/chunk_size+1),"wb+"))



