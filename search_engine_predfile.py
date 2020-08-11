#!/usr/bin/env python

import json
import os

def generate_predfile(search_phrase):
    question = search_phrase

    predfile='{"version": "v2.0", "data": [{"title": "Search Engine", "paragraphs": ['
    for filename in os.listdir("search_engine_contexts"):
        path=os.path.join("search_engine_contexts",filename)
        with open(path) as read_file:
            data=json.load(read_file)
            context=""
            for i in range(len(data['body_text'])):
                context+=data['body_text'][i]['text']
            context=context.replace("\"","'")
            paper_id=data['paper_id']
            predfile+='{"qas" : [{"question":"'+ question + '", "id": "' + str(paper_id) + '", "is_impossible": false}], "context" : "' + context + '"}, '
        read_file.close()
    predfile=predfile[:-2]+']}]}'
    with open("predfile.json","w+") as f1:
        f1.write(predfile)
    f1.close()



if __name__=="__main__":
    search_phrase = input("Search Phrase: ")
    generate_predfile(search_phrase)

