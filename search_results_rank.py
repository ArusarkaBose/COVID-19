#!/usr/bin/env python

import json
import os

def rank():
    with open("data/nbest_predictions.json") as read_file:
        pred=json.load(read_file)

    probabilities=[]
    context_names=[]
    for filename in os.listdir("search_engine_contexts"):
        context_names.append(filename[0:-5])
        probabilities.append(pred[filename[0:-5]][0]['probability'])

    search_result=dict(zip(context_names,probabilities))
    print("")
    print("Results:")
    for (k,v) in sorted(search_result.items(), key=lambda item: item[1], reverse=True):
        print(k)


if __name__ == "__main__":
    rank()
