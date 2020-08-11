#!/usr/bin/env python

import spacy
import scispacy
import neuralcoref

from preprocessing import *
from entity_network import filter_graph
from EntityLinking import pair_table

spacy.prefer_gpu()

filename="pdf_json/0e76160729d01c33942105e505724f59a5db1a44.json"
node="SCoV"

if __name__ == '__main__':
    with open(filename) as f:
            text=""
            for line in f:
                if len(line.lstrip())>50:
                    line=line.lstrip().replace("\"text\":","")
                    line=line.replace("BIBREF","")
                    line=line.replace("ref_id","")
                    line=line.replace("cite_spans","")
                    text+=line.lstrip()
    f.close()

    if(len(text)>=10**6):
        print("Skipping ",count)
        exit() 


    nlp=spacy.load("en_ner_bionlp13cg_md")
    nlp.add_pipe(set_custom_boundaries, before="parser")
    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents)
    neuralcoref.add_to_pipe(nlp)
    doc=nlp(text)

    pairs=pair_table(doc,nlp)
    filter_graph(pairs,node)
