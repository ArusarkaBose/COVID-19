#!/usr/bin/env python

import json
import nltk
import spacy
import scispacy
import pandas as pd

from traindata_prepQA import prep_context

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
tokenizer._params.abbrev_types.add('fig')
tokenizer._params.abbrev_types.add('e.g')

spacy.prefer_gpu()

index=0

def generate_samples():
    with open("dev_data.json") as read_file:
        data=json.load(read_file)
    read_file.close()

    samples=[]

    for paragraph in data['data'][0]['paragraphs']:
        for qas in paragraph['qas']:
            positive_corpus=[]
            negatives=[]
            
            sample_dict=dict()
            
            sample_dict["question"]=qas['question']
            for answers in qas['answers']:
                positive_corpus.append(answers['text'])
            
            positives=[]
            for corpus in positive_corpus:
                for positive in tokenizer.tokenize(corpus):
                    if(len(positive.split())>1):
                        positives.append(positive)
                        
            sample_dict["positives"]=positives
            
            for line in tokenizer.tokenize(paragraph['context']):
                crux=line.replace(" ","")
                if not crux in [pos.replace(" ","") for pos in positives]:
                    negatives.append(line)
                    
            sample_dict["negatives"]=negatives
            
            samples.append(sample_dict)

    return samples




if __name__ == "__main__":
    with open("dev_data.tsv","w+",encoding="utf8") as dev_fh:
        header="Quality	#1 ID	#2 ID	#1 String	#2 String\n"
        dev_fh.write(header)

        samples=generate_samples()
        for sample in samples:
            print("Processing sample ...")

            s1=sample["question"][:-1]+"."

            for positive in sample["positives"]:
                label=1
                s2=positive
                id1=index
                id2=index+1

                index+=2

                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            
            for negative in sample["negatives"]:
                label=0
                s2=negative
                id1=index
                id2=index+1

                index+=2

                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    dev_fh.close()

    print("Resampling ... ")
    df=pd.read_csv("dev_data.tsv",sep="\t",header=0)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("dev_data.tsv", sep="\t")