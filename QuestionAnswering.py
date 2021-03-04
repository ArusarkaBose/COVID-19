#!/usr/bin/env python

print("Importing packages ...")

import os
import json
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.multiprocessing import cpu_count
import torch.multiprocessing as mp
import queue
import torch
import numpy as np
import time

context_length=200

#BERT_Squad_Model='bert-large-uncased-whole-word-masking-finetuned-squad'
#BERT_Squad_Model='bert-base-uncased'
BERT_Squad_Model='distilbert-base-uncased-distilled-squad'
#BERT_Squad_Model='./scibert/scibert_scivocab_uncased_pt'
#BERT_Squad_Model='trained_models/wwm_uncased_finetuned_squad'

def process_file(publication_pool,q,tokenizer,model,question):
    max_score=0 
    best_answer=""

    for filename in publication_pool:
        path=os.path.join("search_engine_contexts",filename)
        with open(path) as read_file:
            data=json.load(read_file)
        read_file.close()
        context_full=""
        for i in range(len(data['body_text'])):
            context_full+=data['body_text'][i]['text']
        context_full=context_full.replace("\"","'")
        paper_id=data['paper_id']
        print("Checking ",paper_id," ...")

        inilist = [m.start() for m in re.finditer(r" ", context_full)]
        part_list=[0]+inilist
        part_list=part_list[::context_length]
        part_list=part_list+[len(context_full)]

        try:
            for i in range(len(part_list)-1):
                context=context_full[part_list[i]:part_list[i+1]]
                context=r'{}'.format(context)

                print("Performing BERT Question-Answering ...")
                with torch.no_grad():
                    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
                    inputs.to('cuda')
                input_ids = inputs["input_ids"].tolist()[0]

                if(len(input_ids)>512):
                    print("Token size exceeded\n")
                    continue

                with torch.no_grad():
                    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    answer_start_scores, answer_end_scores = model(**inputs)
                    answer_start = torch.argmax(answer_start_scores)
                    answer_end = torch.argmax(answer_end_scores) + 1
                    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
                    answer = answer.replace('[CLS] '+question.lower()[:-1]+' ? [SEP]',"")
                    answer = answer.replace('[CLS]',"")
                    answer = answer.replace('[SEP]',"")
                    answer = answer.strip()

                with torch.no_grad():
                    score=(torch.max(answer_start_scores)+torch.max(answer_end_scores)).item()
                if(answer==""):
                    score=-1

                print("Score received =  ",score)

                if(score>max_score):
                    max_score=score
                    best_answer=answer
                del score, answer_start_scores, answer_end_scores

            print("Finished ",paper_id," ...")

            del inputs

        except RuntimeError:
            torch.cuda.empty_cache()
            return

        
    torch.cuda.empty_cache()

    response=(max_score,best_answer)
    q.put(response,block=False)



def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]



if __name__ == "__main__":
    question = input("Search Phrase: ")

    print("Loading SQUAD tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_Squad_Model)
    print("Loading SQUAD model...")
    model = AutoModelForQuestionAnswering.from_pretrained(BERT_Squad_Model)
    model.to('cuda')
    model.share_memory()

    print("Accessing research papers...")    
    procs=cpu_count()
    chunkedPaths = list(chunk(os.listdir('search_engine_contexts'), int(np.ceil(len(os.listdir('search_engine_contexts'))/float(procs/2)))))
    ctx = mp.get_context('spawn')
    q = ctx.Queue()

    processes=[]
    for i in range(len(chunkedPaths)):
        try:
            p=ctx.Process(target=process_file,args=(chunkedPaths[i],q,tokenizer,model,question,))
            processes.append(p)
            p.start()
        except RuntimeError:
            print("Breaking at number of processes = ",len(processes))
            torch.cuda.empty_cache()
            break


    results=[]
    liveprocs = processes
    while liveprocs:
        try:
            while True:
                return_sequence=q.get(block=False)
                results.append(return_sequence)
        except queue.Empty:
            pass

        time.sleep(0.5)   
        if not q.empty():
            continue
        liveprocs = [p for p in liveprocs if p.is_alive()]


    for p in processes:
        p.join()

    while not q.empty():
        return_sequence=q.get(block=False)
        results.append(return_sequence)

    
    best_answer=max(results,key=lambda item:item[0])

    print("\n\n\n\n",best_answer,"\n\n\n\n")


    print("\n\n\n\n")
    with open("QAResults.txt",'w+') as f:
        line=str(best_answer[0])+" : "+str(best_answer[1])
        f.write(line)
    f.close()
    print("Results published!")