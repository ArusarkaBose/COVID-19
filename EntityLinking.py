#!/usr/bin/env python

import os
import spacy
import scispacy
import pandas as pd
import neuralcoref
import numpy as np

from preprocessing import *
from entity_network import draw_graph

spacy.prefer_gpu()



def refine_ent(ent,sent,nlp):
    unwanted_tokens = (
        'PRON',  # pronouns
        'PART',  # particle
        'DET',  # determiner
        'SCONJ',  # subordinating conjunction
        'PUNCT',  # punctuation
        'SYM',  # symbol
        'X',  # other
        )
    ent_type = ent.ent_type_  # get entity type
    if ent_type == '':
        ent_type = 'NOUN_CHUNK'
        ent = ' '.join(str(t.text) for t in
                nlp(str(ent)) if t.pos_
                not in unwanted_tokens and t.is_stop == False)
    return ent,ent_type



def pair_table(doc,nlp):
    ent_pairs=list()
    doc=nlp(doc._.coref_resolved)
    sentences = [sent.text.strip() for sent in doc.sents]

    for sent in sentences:
        sent=nlp(sent)
        spans=list(sent.ents)+list(sent.noun_chunks)
        spans = spacy.util.filter_spans(spans)
        with sent.retokenize() as retokenizer:
            [retokenizer.merge(span) for span in spans]
        for ent in sent.ents:
            token=sent[ent.start]
            subject=[]
            temp_tok=token
            while(True):
                subject = [w for w in temp_tok.head.lefts if w.dep_ in ('subj', 'nsubj','nsubjpass','csubj','csubjpass')]
                if(len(subject)==0 and temp_tok.dep_!='ROOT'):
                    temp_tok=temp_tok.head
                else:
                    break
            

            if(subject):
                subject=subject[0]
                if(token.text==subject.text):
                    continue
                subj_refined, subject_type=refine_ent(subject, sent, nlp)
                ctr=-1
                while((subject.i+ctr)>=0 and subject.nbor(ctr).dep_ in ('amod','punct','nmod')):
                    addition,_=refine_ent(subject.nbor(ctr), sent, nlp)
                    subj_refined=" ".join((str(addition),str(subj_refined)))
                    if(subject.nbor(ctr).i==0):
                        break
                    ctr-=1

                ctr=1
                while((subject.i+ctr)<len(sent) and (subject.nbor(ctr).dep_ in ('nmod','punct','cc','conj') or subject.nbor(ctr).tag_ in ('NN'))):
                    addition,_=refine_ent(subject.nbor(ctr), sent, nlp)
                    subj_refined=" ".join((str(subj_refined),str(addition)))
                    if(subject.nbor(ctr).i==len(sent)-1):
                        break
                    ctr+=1
                subject=subj_refined
            
                flag_val=0
                for ent in sent.ents:
                    ch_list=[[i==k for i in str(subject).split()] for k in ent.text.split()]
                    flag_val+=np.array([i.count(True) for i in ch_list]).sum()
                if(flag_val<1):
                    continue


                relation = token.head
                if relation.nbor(1).pos_ in ('ADP', 'PART'):  
                    relation = ' '.join((str(relation),str(relation.nbor(1))))
                tok_refined, object_type = refine_ent(token, sent, nlp)
            
            
# ENABLE FOR TOKENS WITH CONTEXT

#                 ctr=-1;
#                 while((token.i+ctr)>=0 and token.nbor(ctr).dep_ in ('amod','punct','agent','compound')):
#                     addition,_=refine_ent(token.nbor(ctr), sent, nlp)
#                     tok_refined=" ".join((str(addition),str(tok_refined)))
#                     if(token.nbor(ctr).i==0):
#                         break
#                     ctr-=1

#                 ctr=1
#                 flag=False
#                 while((token.i+ctr)<len(sent) and (token.nbor(ctr).dep_ in ('nmod','punct','cc','conj') or token.nbor(ctr).tag_ in ('NN','IN','CC','CD'))):
#                     for i in range(len(sent.ents)):
#                         if(token.nbor(ctr).text==list(sent.ents)[i].text):
#                             flag=True
#                     if(flag):
#                         break
#                     addition,_=refine_ent(token.nbor(ctr), sent, nlp)
#                     tok_refined=" ".join((str(tok_refined),str(addition)))
#                     if(token.nbor(ctr).i==len(sent)-1):
#                         break
#                     ctr+=1
#                 token=tok_refined
            
            
            
# DISABLE FOR SUBJECT WITH CONTEXT

                flagtrial=False
                for ent in sent.ents:
                    for k in ent.text.split():
                        for i in str(subject).split():
                            if(i==k):
                                subject=i
                                flagtrial=True
                if(flagtrial==False):
                    continue

###########################################        
        
        
                if(str(subject).strip()!=str(token).strip()):
                    ent_pairs.append([str(subject), str(relation), str(token), str(subject_type), str(object_type)])

    filtered_ent_pairs = [sublist for sublist in ent_pairs if not any(str(x) == '' for x in sublist)]
    pairs = pd.DataFrame(filtered_ent_pairs, columns=['Subject', 'Relation', 'Object', 'Subject_type', 'Object_type'])

    return pairs





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
            continue 


        nlp=spacy.load("en_ner_bionlp13cg_md")
        nlp.add_pipe(set_custom_boundaries, before="parser")
        merge_ents = nlp.create_pipe("merge_entities")
        nlp.add_pipe(merge_ents)
        neuralcoref.add_to_pipe(nlp)
        doc=nlp(text)

     
        pairs=pair_table(doc,nlp)  
        draw_graph(pairs,filename)

        print("Processed ",count)
