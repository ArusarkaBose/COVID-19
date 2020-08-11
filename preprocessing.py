#!/usr/bin/env python

import re

def preprocess(text):
    text=re.sub("\"paper_id\":\s*\"\w*\"[,,\n]*","",text)
    text=re.sub("\[\d*\]\s*","",text)
    text=re.sub("\"title\":\s*[,]*","",text)
    text=re.sub("(Fig|fig)\s?\w*\s?","",text)
    text=re.sub("(Figref|figref)\w*\s?","",text)
    text=re.sub("https://[\w\.\/]*\s?","",text)
    return text

def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "\n":
            doc[token.i+1].is_sent_start = True
    return doc
