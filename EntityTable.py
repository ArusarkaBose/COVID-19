#!/usr/bin/env python

import os
import spacy
import scispacy
import pandas as pd

from preprocessing import *

spacy.prefer_gpu()

if __name__ == '__main__':
    df=pd.DataFrame(columns=["PaperId","Text","Entity","Lemma","PaperCount","DatasetCount"])  

    count=0
    for filename in os.listdir("pdf_json"):
        count+=1
        if(count==11):
            break
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
            ff.close()    
    

        nlp=spacy.load("en_ner_bionlp13cg_md")
        nlp.add_pipe(set_custom_boundaries, before="parser")
        doc=nlp(text)

        for sent in doc.sents:
            if(len(sent.ents)):
                for ent in sent.ents:
                    sentence=sent.text
                    paperid=filename[0:-5]
                    entity=ent.text
                    lemma=ent.lemma_
                    papercount=0
                    with open(os.path.join("NER","pdf_json",filename[0:-4]+"txt")) as f:
                        for line in f:
                            if(line.replace(line.split()[-1],"").strip()==lemma):
                                papercount=int(line.split()[-1])
                                break
                    f.close()
                    df_add=k=dict(zip(df.columns,[paperid,sentence,entity,lemma,papercount,0]))
                    df=df.append(df_add,ignore_index=True)


        print("Processed ",count) 

    
    count=0
    for entity in df.Entity:
        count+=1
        mask=df["Entity"]==entity
        print("Entity ",count)
        if(df.loc[mask,"DatasetCount"].all()!=0):
            continue
        dcount=0
        for filename in os.listdir("NER/pdf_json"):
            path=os.path.join("NER/pdf_json",filename)
            with open(path) as f:
                for line in f:
                    if(line.replace(line.split()[-1],"").strip()==df[mask].iloc[0,3]):
                        dcount+=int(line.split()[-1])
                        break
            f.close()
        
        df.loc[mask,"DatasetCount"]=dcount

    df.to_csv("NER/pdf_json.csv")

