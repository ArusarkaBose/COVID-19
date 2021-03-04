#!/usr/bin/env python

print("Importing packages ...")

import os
import json
import re
import nltk
from transformers import BertTokenizer, BertForQuestionAnswering, BertForSequenceClassification
import torch
from torch.multiprocessing import cpu_count
import torch.multiprocessing as mp
import numpy as np
import queue
import time

context_length=200
num_results=10

NLTKtokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
NLTKtokenizer._params.abbrev_types.add('fig')
NLTKtokenizer._params.abbrev_types.add('e.g')

#BERT_Squad_Model='bert-large-uncased-whole-word-masking-finetuned-squad'
BERT_Squad_Model='bert-base-uncased'
#BERT_Squad_Model='trained_models/wwm_uncased_finetuned_squad'

#BERT_MRPC_Model='bert-base-cased-finetuned-mrpc'
BERT_MRPC_Model='bert-base-cased'
#BERT_MRPC_Model='trained_models/mrpc_output'



def process_file(publication_pool,q,tokenizer,model,sim_tokenizer,sim_model,question):
    search_output=dict()
    chosen_answers=dict()

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

        max_prob=0
        best_answer=""
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
                
                text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                answer_start_scores, answer_end_scores = model(**inputs)
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1
                answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
                answer = answer.replace('[CLS] '+question.lower()[:-1]+' ? [SEP]',"")
                answer = answer.replace('[CLS]',"")
                answer = answer.replace('[SEP]',"")
                answer = answer.strip()

                print("Performing BERT Sentence Classification ...")

                probabilities=[]

                if(answer==""):
                    probability=-1

                else:
                    for line in NLTKtokenizer.tokenize(answer):
                        with torch.no_grad():
                            paraphrase = sim_tokenizer.encode_plus(question, line, return_tensors="pt")
                            paraphrase.to('cuda')

                        if(len(paraphrase['input_ids'])>512):
                            print("Token size exceeded\n")
                            continue

                        paraphrase_classification_logits = sim_model(**paraphrase)[0]
                        paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
                        probability = paraphrase_results[1]
                        probabilities.append(probability)

                    probability=max(probabilities)

                print(paper_id," section having probability =  ",probability)

                if(probability>max_prob):
                    max_prob=probability
                    best_answer=answer

            print("Finished ",paper_id," ...")

            probability=max_prob
            search_output[paper_id]=probability
            chosen_answers[paper_id]=best_answer

            del inputs  
        
        except RuntimeError:
            torch.cuda.empty_cache()
            return

    torch.cuda.empty_cache()
    output=sorted(search_output.items(), key=lambda item: item[1], reverse=True)[:num_results]

    responses=[]
    for (k,v) in output:
        responses.append((k,v,chosen_answers[k]))

    q.put(responses,block=False)



def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]



if __name__ == "__main__":
    question = input("Search Phrase: ")
    search_output=dict()
    chosen_answers=dict()

    print("Loading SQUAD tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(BERT_Squad_Model)
    print("Loading SQUAD model...")
    model = BertForQuestionAnswering.from_pretrained(BERT_Squad_Model)
    model.to('cuda')
    model.share_memory()

    print("Loading MRPC tokenizer...")
    sim_tokenizer = BertTokenizer.from_pretrained(BERT_MRPC_Model)
    print("Loading MRPC model...")
    sim_model = BertForSequenceClassification.from_pretrained(BERT_MRPC_Model)
    sim_model.to('cuda')
    sim_model.share_memory()

    print("Accessing research papers...")
    procs=cpu_count()
    chunkedPaths = list(chunk(os.listdir('search_engine_contexts'), int(np.ceil(len(os.listdir('search_engine_contexts'))/float(procs/2)))))
    ctx = mp.get_context('spawn')
    q = ctx.Queue()

    processes=[]
    for i in range(len(chunkedPaths)):
        try:
            p=ctx.Process(target=process_file,args=(chunkedPaths[i],q,tokenizer,model,sim_tokenizer,sim_model,question,))
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
                for sequence in return_sequence:
                    results.append(sequence)
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
        for sequence in return_sequence:
            results.append(sequence)


    top_results=sorted(results,key=lambda item:item[1],reverse=True)[:num_results]

    print("\n\n\n\n")
    with open("SearchResults.txt",'w+') as f:
        for (k,v,e) in top_results:    
            f.write(k+" : "+e+"\n")
            f.write("\n")
    f.close()
    print("Results published!")
