#!/usr/bin/env python

import os
import spacy
import scispacy
import re
from spacy import displacy

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
            with open(os.path.join("DepParse","remaining.txt"),"a+") as ff:
                ff.write(path+"\n")
                continue
            ff.close()

        
        nlp=spacy.load("en_ner_bionlp13cg_md")
        nlp.add_pipe(set_custom_boundaries, before="parser")
        doc=nlp(text)

        
        options={"compact":True,"collpase_phrases":True, "distance":110,"fine_grained":True}
       

        html=""
        for sent in list(doc.sents):
            if(len(sent.ents)>0):
                html+=displacy.render(sent,style="dep",minify=True,options=options,page=True,jupyter=False)
                html+=displacy.render(sent,style="ent",minify=True,options=options,page=True,jupyter=False)



        with open(os.path.join("DepParse","pdf_json",filename[0:-4]+"html"),"w+") as f1:
            f1.write(html)
        f1.close()

        print("Processed ",count)   

