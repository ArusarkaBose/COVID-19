#!/usr/bin/env python

print("Importing packages ...")

import os
import json
import re
from transformers import BertTokenizer, BertForQuestionAnswering, BertForSequenceClassification
import torch

context_length=200

#BERT_Squad_Model='bert-large-uncased-whole-word-masking-finetuned-squad'
BERT_Squad_Model='bert-base-uncased'

BERT_MRPC_Model='bert-base-cased-finetuned-mrpc'

def main():
    question = input("Search Phrase: ")
    search_output=dict()
    chosen_answers=dict()

    print("Loading SQUAD tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(BERT_Squad_Model)
    print("Loading SQUAD model...")
    model = BertForQuestionAnswering.from_pretrained(BERT_Squad_Model)
    model.to('cuda')

    print("Loading MRPC tokenizer...")
    sim_tokenizer = BertTokenizer.from_pretrained(BERT_MRPC_Model)
    print("Loading MRPC model...")
    sim_model = BertForSequenceClassification.from_pretrained(BERT_MRPC_Model)
    sim_model.to('cuda')

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

        max_prob=0
        best_answer=""
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

            print("Performing BERT Sentence Classification ...")
            paraphrase = sim_tokenizer.encode_plus(question, answer, return_tensors="pt")
            paraphrase.to('cuda')
            paraphrase_classification_logits = sim_model(**paraphrase)[0]
            paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
            probability = paraphrase_results[1]

            if(answer==""):
                probability=-1

            print(paper_id," section having probability =  ",probability)

            if(probability>max_prob):
                max_prob=probability
                best_answer=answer

        print("Finished ",paper_id," ...")

        probability=max_prob
        search_output[paper_id]=probability
        chosen_answers[paper_id]=best_answer

    print("\n\n\n\n")
    with open("SearchResults.txt",'w+') as f:
        for (k,v) in sorted(search_output.items(), key=lambda item: item[1], reverse=True):
            f.write(k+" : "+chosen_answers[k]+"\n")
            f.write("\n")
    f.close()
    print("Results published!")


if __name__ == "__main__":
    main()