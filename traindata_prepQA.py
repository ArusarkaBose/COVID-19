#!/usr/bin/env python
import json
import os
import re
import spacy
import scispacy
from nltk.stem import WordNetLemmatizer 
from scispacy.abbreviation import AbbreviationDetector

lemmatizer = WordNetLemmatizer() 

id_count=0

keywords=['structure','symptoms','medicine','regions','surfaces']
keyword_links={
    'structure': ["structure","constituents","composition"],
    'symptoms': ["symptoms","effects","diseases"],
    'medicine': ["medicine","chemical","vaccine","antidote"],
    'regions': ["regions","places","countries","states"],
    'surfaces': ["surfaces"]
}

QUERY_LIST=[
    "Which vaccines are available?",
    "Structure of coronavirus?",
    "Which chemicals can be used to kill the virus?",
    "Which regions have been affected with COVID?",
    "What can be the effects of COVID-19?",
    "What diseases are associated with Coronavirus?",
    "Medicines to treat COVID?",
    "Vaccines for COVID?",
    "Is there any antidote?",
    "On which surfaces does the virus survive?",
    "Which places have been affected with the virus?",
    "What symptoms occur after contracting COVID-19?",
    "What is the structure of the virus?",
    "Structure?",
    "Symptoms?",
    "Vaccines?",
    "Regions affected?",
    "Antidote?",
    "Is there any medicine for COVID?",
    "Coronavirus structure?",
    "COVID symptoms?",
    "Symptoms of COVID?",
    "Structure of COVID?",
    "Most affected regions?",
    "Effects of COVID-19?",
    "Affected places?",
    "Diseases due to COVID virus?",
    "Virus composition?",
    "Constituents of the virus?",
    "Composition of coronavirus?",
    "Virus structure?",
    "COVID structure?",
    "Coronavirus composition?",
    "Cellular composition of the virus?",
    "Virus cell structure?",
    "Cell structure of virus?",
    "Coronavirus symptoms?",
    "Virus DNA structure?",
    "Virus DNA composition?",
    "Constituents of virus DNA?",
    "COVID vaccines?",
    "Surfaces on which the virus can survive?",
    "Surfaces on which coronavirus will survive?",
    "What can be the symptoms of COVID?",
    "What symptoms mean COVID-19?"
]

spacy.prefer_gpu()

def prep_context():
    context_length=200
    contexts=[]

    for filename in os.listdir("search_engine_contexts"):
        path=os.path.join("search_engine_contexts",filename)
        with open(path) as read_file:
            data=json.load(read_file)
        read_file.close()
        context_full=""
        for i in range(len(data['body_text'])):
            context_full+=data['body_text'][i]['text']
        context_full=context_full.replace("\"","'")

        inilist = [m.start() for m in re.finditer(r" ", context_full)]
        part_list=[0]+inilist
        part_list=part_list[::context_length]
        part_list=part_list+[len(context_full)]

        for i in range(len(part_list)-1):
            context=context_full[part_list[i]:part_list[i+1]]
            context=r'{}'.format(context)
            contexts.append(context)

    return contexts



def generate_answers(keyword,context):
    answers=[]
    start_indices=[]
    nlp = spacy.load("en_core_sci_lg")

    abbreviation_pipe = AbbreviationDetector(nlp)
    nlp.add_pipe(abbreviation_pipe)
       
    doc=nlp(context)
    print("Choose indices : ")
    print(list(enumerate([ent._.long_form if(ent._.long_form!=None) else ent for ent in doc.ents])))
    print("\n\n\n")
    
    print("Keyword = : ",keyword,"\n")
    gold_indices = [int(item) for item in input("Enter indices : ").split()]
    print("\n\n")
    try:
        gold_ents=[doc.ents[i] for i in gold_indices]
    except IndexError:
        print("Invalid indices")
        print("Skipping context ... ")
        return (answers,start_indices)

    
    with doc.retokenize() as retokenizer:
        [retokenizer.merge(span) for span in gold_ents]

    subtrees=[]
    for token in doc:
        if token.text in [ent.text for ent in gold_ents]:
            root=token
            while(root.head!=root):
                root=root.head
            subtree=root.subtree
            subtree=list(subtree)
            start_index=subtree[0].idx
            if(len(subtrees)!=0):
                if(subtrees[-1][-1].i==subtree[0].i-1):
                    subtrees[-1]+=subtree
                    continue
            if(len(subtrees)==0):
                subtrees.append(subtree)
                start_indices.append(start_index)
            elif(not (any(subtree == subtrees[-1][i:i + len(subtree)] for i in range(len(subtrees[-1])-len(subtree) + 1)))):
                subtrees.append(subtree)
                start_indices.append(start_index)

    for subtree in subtrees:
        sentence=" ".join([word.text for word in subtree])
        answers.append(sentence)

    return(answers,start_indices)



def collect_queries():
    queries=[]
    print("Enter -1 when you have no more queries")
    while(True):
        query=input("Enter : ")
        if(query=="-1"):
            break
        queries.append(query)
    return queries



def get_keywords(query):
    choices=[0]*len(keywords)
    query_txt=query.lower()
    query_txt=" ".join([lemmatizer.lemmatize(word) for word in query_txt.split()])
    
    for keyword in keywords:
        mask=[re.search(lemmatizer.lemmatize(word),query_txt)!=None for word in keyword_links[keyword]]
        if sum(mask):
            choices[keywords.index(keyword)]=1
    key_list=[d for d, s in zip(keywords, choices) if s]
    return key_list



def create_db():
    database=[]

    contexts=prep_context()
    for context in contexts:
        db_dict=dict()
        for keyword in keywords:
            db_dict[keyword]=generate_answers(keyword,context)
        database.append(db_dict)
    return (contexts,database)



if __name__ == "__main__":
    try:
        with open("dev_data.json") as read_file:
            dev_file=json.load(read_file)
        read_file.close()
        print("Previously created training sample exists. Appending new data to existing file ...")

        print("Preparing contexts and database ... ")
        contexts,database=create_db()
        print("Collecting queries ... ")
        #queries=collect_queries()                        # Uncomment to use custom queries for setting up samples
        queries=QUERY_LIST
        print("Collected ... ")

        for context in contexts:
            paragraph_dict=dict()
            paragraph_dict["qas"]=""
            paragraph_dict["context"]=context

            qas_dict_list=[]
            for query in queries:
                qas_dict=dict()
                qas_dict["question"]=query
                id_count+=1
                qas_dict["id"]=str(id_count)
                qas_dict["answers"]=""
                qas_dict["is_impossible"]="false"

                answers_dict_list=[]
                key_list=get_keywords(query)

                answers_dict=dict()
                for key in key_list:
                    text_list,answer_start_list=database[contexts.index(context)][key]

                    for text,answer_start in zip(text_list,answer_start_list):
                        answers_dict["text"]=text
                        answers_dict["answer_start"]=answer_start
                        answers_dict_list.append(answers_dict.copy())

                qas_dict["answers"]=answers_dict_list
                qas_dict_list.append(qas_dict.copy())
            
            paragraph_dict["qas"]=qas_dict_list
            dev_file["data"][0]["paragraphs"].append(paragraph_dict.copy())

        print("Writing to dev_data.json ...")
        with open("dev_data.json","w+") as write_file:
            json.dump(dev_file,write_file)
        write_file.close()


    except(FileNotFoundError):
        print("Existing file not found. Creating new samples ...")
        dev_file=dict()
        dev_file["version"]="v2.0"
        dev_file["data"]=""
        
        data_dict=dict()
        data_dict["title"]='SearchEngine'
        data_dict["paragraphs"]=""
        
        paragraph_dict_list=[]

        print("Preparing contexts and database ... ")
        contexts,database=create_db()
        print("Collecting queries ... ")
        #queries=collect_queries()                        # Uncomment to use custom queries for setting up samples
        queries=QUERY_LIST
        print("Collected ... ")

        for context in contexts:
            paragraph_dict=dict()
            paragraph_dict["qas"]=""
            paragraph_dict["context"]=context

            qas_dict_list=[]
            for query in queries:
                qas_dict=dict()
                qas_dict["question"]=query
                id_count+=1
                qas_dict["id"]=str(id_count)
                qas_dict["answers"]=""
                qas_dict["is_impossible"]="false"

                answers_dict_list=[]
                key_list=get_keywords(query)

                answers_dict=dict()
                for key in key_list:
                    text_list,answer_start_list=database[contexts.index(context)][key]

                    for text,answer_start in zip(text_list,answer_start_list):
                        answers_dict["text"]=text
                        answers_dict["answer_start"]=answer_start
                        answers_dict_list.append(answers_dict.copy())

                qas_dict["answers"]=answers_dict_list
                qas_dict_list.append(qas_dict.copy())
            
            paragraph_dict["qas"]=qas_dict_list
            paragraph_dict_list.append(paragraph_dict.copy())

        data_dict["paragraphs"]=paragraph_dict_list
        dev_file["data"]=[data_dict.copy()]


        with open("dev_data.json","w+") as write_file:
            json.dump(dev_file,write_file)
        write_file.close()