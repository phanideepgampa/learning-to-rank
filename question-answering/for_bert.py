import os
data = "wiki_qa"

sets = ["train","test","dev"]

for s in sets:
    f= open(os.path.join(data,s+"-bert.txt"),"w",encoding='utf-8')
    with open(os.path.join("wiki_qa/"+s,"a.toks")) as questions,open(os.path.join("wiki_qa/"+s,"b.toks")) as answers:
        for q,a in zip(questions,answers):
            line=" ".join((q.strip(),"|||",a.strip()))
            f.write(line+"\n")
    f.close()
