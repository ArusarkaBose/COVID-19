#!/usr/bin/env python

print("Importing packages ...")

import os
import json
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

context_length=200

#BERT_Squad_Model='bert-large-uncased-whole-word-masking-finetuned-squad'
#BERT_Squad_Model='bert-base-uncased'
#BERT_Squad_Model='distilbert-base-uncased-distilled-squad'
BERT_Squad_Model='./scibert/scibert_scivocab_uncased_pt'

def main():
    question = input("Search Phrase: ")

    print("Loading SQUAD tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_Squad_Model)
    print("Loading SQUAD model...")
    model = AutoModelForQuestionAnswering.from_pretrained(BERT_Squad_Model)
    model.to('cuda')

    max_score=0 
    best_answer=""
    print("Accessing research papers...")
    for filename in os.listdir("search_engine_contexts"):
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

        for i in range(len(part_list)-1):
            context=context_full[part_list[i]:part_list[i+1]]
            context=r'{}'.format(context)

            print("Performing BERT QUestion-Answering ...")
            inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
            inputs.to('cuda')
            input_ids = inputs["input_ids"].tolist()[0]
            text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = model(**inputs)
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            answer = answer.replace('[CLS] '+question.lower()[:-1]+' ? [SEP]',"")
            answer = answer.replace('[CLS]',"")
            answer = answer.replace('[SEP]',"")
            answer = answer.strip()

            score=torch.max(answer_start_scores)+torch.max(answer_end_scores)
            if(answer==""):
                score=-1

            print("Score received =  ",score)

            if(score>max_score):
                max_score=score
                best_answer=answer

        print("Finished ",paper_id," ...")

    print("\n\n\n\n")
    with open("QAResults.txt",'w+') as f:
        f.write(best_answer)
    f.close()
    print("Results published!")


if __name__ == "__main__":
    main()