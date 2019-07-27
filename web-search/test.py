unique=[]
prev = -1
with open("MQ2007/Fold1/vali.txt") as data:
     for line in data:
             qid = int(line.split(' ')[1].split(':')[1])
             if prev != qid :
                prev =qid 
                if qid in unique:
                    print("failure")
                    break
                else:
                    unique.append(qid)
print(len(unique))
