#!/usr/bin/env python

import os
import spacy
import scispacy
from collections import OrderedDict
from tabulate import tabulate

from preprocessing import *

spacy.prefer_gpu()

if __name__ == '__main__':
    count=0
    for filename in os.listdir("pdf_json"):
        count+=1
        path=os.path.join("pdf_json",filename)
        with open(path) as f:
            text=""
            for line in f:
                if len(line.lstrip())>50:
                    line=line.lstrip().replace("\"text\":","")
                    line=line.replace("BIBREF","")
                    line=line.replace("ref_id","")
                    line=line.replace("cite_spans","")
                    text+=line.lstrip()
        f.close()

        text=preprocess(text)

        if(len(text)>=10**6):
            print("Skipping ",count)
            with open(os.path.join("NER","remaining.txt"),"a+") as ff:
                ff.write(path+"\n")
                continue
            ff.close()    
    

        nlp=spacy.load("en_ner_bionlp13cg_md")
        nlp.add_pipe(set_custom_boundaries, before="parser")
        doc=nlp(text)
    

        entities=[]
        for i in range(len(doc.ents)):
            if(doc.ents[i].root.is_stop!=True):
                entities.append(doc.ents[i].lemma_)
    

        table=[]
        for i in list(OrderedDict.fromkeys(entities)):
            table.append((i,entities.count(i)))


        with open(os.path.join("NER","pdf_json",filename[0:-4]+"txt"),"w+") as f1:
            f1.write(tabulate(table,headers=["Named Entity","Count"]))
        f1.close()

        print("Processed ",count)    
