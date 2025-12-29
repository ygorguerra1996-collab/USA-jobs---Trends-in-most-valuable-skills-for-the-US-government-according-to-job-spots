### Middle Stage: Generate a list or dict with the top-n ngrams and after data come again from the start,  and if the n gram matches
### with any of the top n-grams list/dict, send the n-gram to the db and cut from bloco_descricao

import requests #API
import sqlite3 #db
import time 
from datetime import datetime
from bs4 import BeautifulSoup #html parser
import spacy #NLP
import unicodedata #module that allows Unicode char manipulation 
import pandas as pd
import math
from spacy.lang.en.stop_words import STOP_WORDS #its gonna help in stage 2. to filter stopwords, not having to use doc.
import pickle
import os

# we're gonna need to connect to the db again since we divided the stages

conn = sqlite3.connect('USAjobs.db')
cursor = conn.cursor()

LOAD_DIR = "C:\\Users\\ygorg\\OneDrive\\Documentos\\Pickles for remotive project"
path = os.path.join(LOAD_DIR, "stage1_data.pkl")

with open (path, "rb") as f: #calling the file, now reading binary "rb", opening it to get the vars values from stage 1
    stage1_data = pickle.load(f) #placing the dict of the objects in this var. Bellow we're going to reference the values of each value,
    #just so we have 'em also here in stage 2

bigrams = stage1_data["bigrams"]
trigrams = stage1_data["trigrams"]
fourgrams = stage1_data ["fourgrams"]
fivegrams = stage1_data ["fivegrams"]
ngrams_to_be_cut = stage1_data ["ngrams_to_be_cut"]
bloco_de_descricao = stage1_data ["bloco_de_descricao"]


dict_bigrams_rank = {}
dict_trigrams_rank = {}
dict_fourgrams_rank = {}
dict_fivegrams_rank = {}

dict_jobid_index_to_remove_from_bloco_stage2 = {}
jobid_on_keys_for_performance_first = {}

bigramslist = []
trigramslist = []
fourgramslist = []
fivegramslist = []

#already optimized the beggining of this stage, that was causing trouble

print ("Starting to create optimized bloco descrição - Jobid on keys for performance")
start = time.time()

for d in bloco_de_descricao:
    jobid_on_keys_for_performance_first.setdefault(d["jobid"],[]).append(d)

end = time.time()
print (f"finished creating the optimized bloco descricao. Time required: {end-start:.2f} seconds")


print (" starting to separate the words in jd in bi,tri,four,five grams, matching with the excel and creating the ranks dicts and refreshing bloco de descricao ")
start = time.time()

negatives_to_machine_learning_stg3 = set()

for jobid in jobid_on_keys_for_performance_first:
    bloco_vaga_stg2 = jobid_on_keys_for_performance_first[jobid] #only the jobid objects (dicts ) from bloco descricao

    only_words_stg2 = [d["word"] for d in bloco_vaga_stg2] #has only the words of the job.
        #agora preciso dividir as palavras em bi,tri,four,five.
        #tenho que dar um jeito de anexar os indexes, junto.
    only_indexes_stg2 = [d["index"] for d in bloco_vaga_stg2]

    bigramslist, trigramslist, fourgramslist, fivegramslist = [], [], [], [] #cleansing to prevent data leak in the second iteration
    
    for i in range (len(only_words_stg2)-1):
        bigramslist.append((only_words_stg2[i:i+2],[only_indexes_stg2[i],only_indexes_stg2[i+1]]))#tuple with the gram and a list of indexes
    for i in range (len(only_words_stg2)-2):
        trigramslist.append((only_words_stg2[i:i+3],[only_indexes_stg2[i],only_indexes_stg2[i+1],only_indexes_stg2[i+2]]))
    for i in range (len(only_words_stg2)-3):
        fourgramslist.append((only_words_stg2[i:i+4],[only_indexes_stg2[i],only_indexes_stg2[i+1],only_indexes_stg2[i+2],only_indexes_stg2[i+3]]))
    for i in range (len(only_words_stg2)-4):
        fivegramslist.append((only_words_stg2[i:i+5],[only_indexes_stg2[i],only_indexes_stg2[i+1],only_indexes_stg2[i+2],only_indexes_stg2[i+3],only_indexes_stg2[i+4]]))
        
        #I'm filling the lists with grams. So that later I can compare them with the excel.


    for ngrams_list in [bigramslist,trigramslist,fourgramslist,fivegramslist]:
        for ngrams, indexes in ngrams_list:
            clean_gram = ' '.join(ngrams)
            if clean_gram in ngrams_to_be_cut:
                indexes_to_removestg2 = set(indexes)

                negatives_to_machine_learning_stg3.add(clean_gram) #cause update expects an iterable

    
                if dict_jobid_index_to_remove_from_bloco_stage2.get(jobid) is None:
                    dict_jobid_index_to_remove_from_bloco_stage2[jobid] = set(indexes_to_removestg2)
    
                else: dict_jobid_index_to_remove_from_bloco_stage2[jobid].update(indexes_to_removestg2)

                continue

        
        # a primeira parte esta bacana, economia de performance imensa. O gpt falou em 1000x menos. Agora precisamos só encaixar o restante,
        #que cria a contagem por gram. Que não vai ser dificil. Só fica ligeiro pq estamos limpando eles a cada novo jobid, la em cima,
        #mas isso não vai atrapalhar se já criarmos o dict rank a cada iteração. (ele cria antes de zerar os dicts)

            if len(ngrams) == 2:

                if dict_bigrams_rank.get (clean_gram) is None:
                    dict_bigrams_rank[clean_gram] = 1
                else: 
                    dict_bigrams_rank[clean_gram] = dict_bigrams_rank[clean_gram]+1
        
            if len(ngrams) == 3:

                if dict_trigrams_rank.get(clean_gram) is None:
                    dict_trigrams_rank[clean_gram] =1
                else: dict_trigrams_rank[clean_gram] = dict_trigrams_rank[clean_gram] + 1

            if len(ngrams) == 4:

                if dict_fourgrams_rank.get(clean_gram) is None:
                    dict_fourgrams_rank[clean_gram] = 1
                else: dict_fourgrams_rank[clean_gram] = dict_fourgrams_rank[clean_gram] + 1

            if len(ngrams) == 5:

                if dict_fivegrams_rank.get(clean_gram) is None:
                    dict_fivegrams_rank[clean_gram] = 1
                else: 
                    dict_fivegrams_rank[clean_gram] = dict_fivegrams_rank[clean_gram] + 1

bloco_de_descricao[:] = [d for d in bloco_de_descricao if not (d["jobid"] in dict_jobid_index_to_remove_from_bloco_stage2 and 
d["index"] in dict_jobid_index_to_remove_from_bloco_stage2[d["jobid"]])] #removing from bloco_descricao jobids and indexes that are related
#to common words that are not skills.

end = time.time()
print (f"finished last task. Time required: {end-start:.2f} seconds")


print ("Started cutting <30 and < 15 grams")
start = time.time()

#now, removing grams without limit of 30 appearances (bi and tri, according to rule). The limit for bi and tri is 30 and for the others, 15

for ngrams in [dict_bigrams_rank,dict_trigrams_rank]:
    to_remove = [gram for gram,count in ngrams.items() if count <30]
    for gram in to_remove:
        ngrams.pop(gram)

for ngrams in [dict_fourgrams_rank,dict_fivegrams_rank]:
    to_remove = [gram for gram,count in ngrams.items() if count <15]
    for gram in to_remove:
        ngrams.pop(gram) #limit is filtered. now need to get top 20% of each gram dict.

end = time.time()
print (f"finished last task. Time required: {end-start:.2f} seconds")



print (f"Started sorting the rank dicts by the highest to the lowest count, and cutting only top 20% ")    
start = time.time()

#picking up 20% top bigrams

sorted_bigrams = sorted (dict_bigrams_rank.items(), key= lambda x: x[1] , reverse = True) #returns tuples, sorting from the second parameter
#lambda x: x[1] from the dict. with reverse order. "reverse = true". So, will return tuples sorted Desc based in the count.
rankn_bigrams = max(1,int(len(sorted_bigrams)*0.2)) #we're fixing the position that corresponds to the top 20% rank of bigrams based on their count.
#while len is returning the total grams in the sorted_bigrams, where multiplying by 20% to find the last position, the "1" here is to prevent
#it to return zero if the list is too small
dict_bigrams_rank = [bigram for bigram, count in sorted_bigrams [:rankn_bigrams]] # catching only the bigrams that correspond to the 20% and
#placing in this list, as well as their count. We're saying that we want only until the rankn position. That will correspond to the position 
#that represents the top 20%. #replacing everything that was in the dict_bigrams_rank. As it will serve later as comparison for db insertion.

#picking up 20% top tri

sorted_trigrams = sorted(dict_trigrams_rank.items(), key = lambda x: x[1], reverse=True)
rankn_trigrams = max(1, int(len(sorted_trigrams)*0.2))
dict_trigrams_rank = [trigram for trigram, count in sorted_trigrams [:rankn_trigrams]]

#picking up 20% top fourgrams

sorted_fourgrams = sorted(dict_fourgrams_rank.items(), key = lambda x:x[1],reverse=True)
rankn_fourgrams = max(1,int(len(sorted_fourgrams)*0.2))
dict_fourgrams_rank = [fourgram for fourgram, count in sorted_fourgrams [:rankn_fourgrams]]

#picking up 20% top fivegrams

sorted_fivegrams = sorted(dict_fivegrams_rank.items(),key=lambda x:x[1], reverse=True)
rankn_fivegrams = max (1,int(len(sorted_fivegrams)*0.2))
dict_fivegrams_rank = [fivegram for fivegram,count in sorted_fivegrams [:rankn_fivegrams]]


#nice, the rank dicts are clean with only the top 20%. Now we need to create the logic to go through the bloco descricao creating 
# all the ngrams with the words left, and if they are present in the top 20% rank, send to the db and cut from bloco descricao
# now i need to create a new dict words per id (as I did in the stage 1 so later we can create the ngrams lists only for the same id)
#I'm gonna have to remove stopwords here, or its going to impact in the topn20%

end = time.time()
print (f"finished last task. Time required: {end-start:.2f} seconds")

print (f"Started creating dict words per id stg2")
start = time.time()

dict_words_per_id_stg2 = {} #gonna be used to store description words underneath their jid
for words_dict_stage2 in bloco_de_descricao:

    if words_dict_stage2["word"] not in STOP_WORDS:

       if dict_words_per_id_stg2.get(words_dict_stage2["jobid"]) is None:
            dict_words_per_id_stg2[words_dict_stage2["jobid"]] = [words_dict_stage2["word"]]

       else:
            dict_words_per_id_stg2[words_dict_stage2["jobid"]].append(words_dict_stage2["word"])
end = time.time()
print (f"finished last task. Time required: {end-start:.2f} seconds")

print("started creating ngrams lists again - maybe room for improvement here")
start = time.time()
    
bigrams_stage2 = [] #same thing from stage1 but we have to do it again cause bloco_de_descricao changed since first stage
trigrams_stage2 = []
fourgrams_stage2 = []
fivegrams_stage2 = []

words_to_db_max = 5000
words_to_db = []

for jobid, words in dict_words_per_id_stg2.items():
    for i in range (len(words)-1):
        bigrams_stage2.append((jobid,[words[i],words[i+1]]))

    for i in range (len(words)-2):
        trigrams_stage2.append((jobid,[words[i],words[i+1],words[i+2]]))

    for i in range (len(words)-3):
        fourgrams_stage2.append((jobid,[words[i],words[i+1],words[i+2],words[i+3]]))

    for i in range (len(words)-4):
        fivegrams_stage2.append((jobid,[words[i],words[i+1],words[i+2],words[i+3],words[i+4]]))

end = time.time()
print (f"finished last task. Time required: {end-start:.2f} seconds")

#had to recreate the ngrams lists due to the fact bloco_de_descricao changed since stage1. so now I am considering only the grams that
#havent been cut since stage 1 to avoid duplicates

#now I got the 20% top ngrams in the dicts, and the ngrams in the lists from the bloco_de_descricao.

print ("started creating the optimized blocodescricao - jobid on keys performance for the second time. Makes sense - dont worry")
start = time.time()

dict_jobid_index_to_remove_from_bloco_stage2 = {} #gonna be used to store the jobid and words to be deleted from bloco descricao. Just like
#I did in stage 1.

jobid_on_keys_for_performance = {} #I'm using this one to put the jobid in the keys of the dict, so performance goes to O(1). quicker to 
#for python to analyse. Set default is checking if the key exists, if not, creates and puts the default value. Set default allows you to 
#make the append in the same line. Which is great. Its like a "get" but improved
for d in bloco_de_descricao:
    jobid_on_keys_for_performance.setdefault(d["jobid"],[]).append(d)  
# at the end of this, we're gonna have {123:[{jobid:123, word:grow, index:5}, {jobid:123, word:better, index:6}], 124:[{}]}
#so we can acess directly by the jobid. 

print (f"finished last task. Time required: {end-start:.2f} seconds")

print ("started matching grams with top 20percent grams and sending to db if positive")
start = time.time()

for ngrams in [bigrams_stage2,trigrams_stage2,fourgrams_stage2,fivegrams_stage2]:
    for jobid, grams in ngrams:
        clean_gram = ' '.join(grams).lower()
        
        indexes_to_removestg2 = set()
       

#from here until line 494 I'm doing exactly the same logic that I've done for stage 1 on removing words that are being sent to the db
#from the bloco_de_descricao. I could have done a def for that but I aint got time. So I'm just applying the logic for bi,tri,four and fivegrams
        if len(grams) == 2:
            if clean_gram in dict_bigrams_rank:
                indexes_to_removestg2 = set()
                print (f" Stage2: jobid,Word queued for commit in the db:", jobid,clean_gram)
                words_to_db.append((jobid,clean_gram,'stage 2'))

                bloco_vaga_stg2 = jobid_on_keys_for_performance[jobid]

                for i in range(len(bloco_vaga_stg2)-len(grams)+1):
 
                    slice_words_stage2 = [d["word"] for d in bloco_vaga_stg2 [i:i+len(grams)]]#not going to make three words because
                    #python indexes list goes for example, [0:2] that means until 2 but not including. So only 2 words.


                    if slice_words_stage2 == grams:
                        indexes_to_removestg2.update (d["index"] for d in bloco_vaga_stg2[i:i+len(grams)])

                        if dict_jobid_index_to_remove_from_bloco_stage2.get(jobid) is None: #build this one to remove the indexes in the jobdesc. 
                #that correspond to the ngram that is related to common words (not skill). So we've got bloco descriçao clean for stage3
                            dict_jobid_index_to_remove_from_bloco_stage2[jobid] = set(indexes_to_removestg2) 
                        else: dict_jobid_index_to_remove_from_bloco_stage2[jobid].update(indexes_to_removestg2)
            #agora ta faltando só eu bolar uma forma de optimizar a performance da atualização do bloco descricao. encaixando essa remoção tbm



        if len(grams) == 3:
            if clean_gram in dict_trigrams_rank:
                indexes_to_removestg2 = set()
                print (f" Stage2: jobid,Word queued for commit in the db:", jobid,clean_gram)
                words_to_db.append((jobid,clean_gram,'stage 2'))

                bloco_vaga_stg2 = jobid_on_keys_for_performance[jobid]

                for i in range(len(bloco_vaga_stg2)-len(grams)+1):
 
                    slice_words_stage2 = [d["word"] for d in bloco_vaga_stg2 [i:i+len(grams)]]


                    if slice_words_stage2 == grams:
                        indexes_to_removestg2.update (d["index"] for d in bloco_vaga_stg2[i:i+len(grams)])

                        if dict_jobid_index_to_remove_from_bloco_stage2.get(jobid) is None: #build this one to remove the indexes in the jobdesc. 
                #that correspond to the ngram that is related to common words (not skill). So we've got bloco descriçao clean for stage3
                            dict_jobid_index_to_remove_from_bloco_stage2[jobid] = set(indexes_to_removestg2) 
                        else: dict_jobid_index_to_remove_from_bloco_stage2[jobid].update(indexes_to_removestg2)
           




        if len(grams) == 4:
                if clean_gram in dict_fourgrams_rank:
                    indexes_to_removestg2 = set()
                    print (f" Stage2: jobid,Word queued for commit in the db:", jobid,clean_gram)
                    words_to_db.append((jobid,clean_gram,'stage 2'))

                    bloco_vaga_stg2 = jobid_on_keys_for_performance[jobid]

                    for i in range(len(bloco_vaga_stg2)-len(grams)+1):
 
                        slice_words_stage2 = [d["word"] for d in bloco_vaga_stg2 [i:i+len(grams)]]


                        if slice_words_stage2 == grams:
                            indexes_to_removestg2.update (d["index"] for d in bloco_vaga_stg2[i:i+len(grams)])

                            if dict_jobid_index_to_remove_from_bloco_stage2.get(jobid) is None: #build this one to remove the indexes in the jobdesc. 
                #that correspond to the ngram that is related to common words (not skill). So we've got bloco descriçao clean for stage3
                                dict_jobid_index_to_remove_from_bloco_stage2[jobid] = set(indexes_to_removestg2) 
                            else: dict_jobid_index_to_remove_from_bloco_stage2[jobid].update(indexes_to_removestg2)




        if len(grams) == 5:
            if clean_gram in dict_fivegrams_rank:
                indexes_to_removestg2 = set()
                print (f" Stage2: jobid,Word queued for commit in the db:", jobid,clean_gram)
                words_to_db.append((jobid,clean_gram,'stage 2'))
                bloco_vaga_stg2 = jobid_on_keys_for_performance[jobid]
                for i in range(len(bloco_vaga_stg2)-len(grams)+1):

                    slice_words_stage2 = [d["word"] for d in bloco_vaga_stg2 [i:i+len(grams)]]
                    if slice_words_stage2 == grams:
                        indexes_to_removestg2.update (d["index"] for d in bloco_vaga_stg2[i:i+len(grams)])

                        if dict_jobid_index_to_remove_from_bloco_stage2.get(jobid) is None: #build this one to remove the indexes in the jobdesc. 
                #that correspond to the ngram that is related to common words (not skill). So we've got bloco descriçao clean for stage3
                            dict_jobid_index_to_remove_from_bloco_stage2[jobid] = set(indexes_to_removestg2) 
                        else: dict_jobid_index_to_remove_from_bloco_stage2[jobid].update(indexes_to_removestg2)
    
    if len(words_to_db) >= words_to_db_max:

        cursor.executemany ('''
        INSERT OR IGNORE INTO wordsindescription(jobid, word, stage_origin)
        VALUES (?,?,?)''',words_to_db)
        conn.commit()
        words_to_db = []

if words_to_db:
    cursor.executemany('''
    INSERT OR IGNORE INTO wordsindescription(jobid, word, stage_origin)
    VALUES (?,?,?)''',words_to_db)

end = time.time()
print (f"finished last task. Time required: {end-start:.2f} seconds")

print ("started refreshing bloco de descricao")
start = time.time()

bloco_de_descricao[:] = [d for d in bloco_de_descricao if not (d["jobid"] in dict_jobid_index_to_remove_from_bloco_stage2 and d["index"]
in dict_jobid_index_to_remove_from_bloco_stage2[d["jobid"]])] 

end = time.time()
print (f"finished last task. Time required: {end-start:.2f} seconds")

#segundo o gpt até aqui tudo certo, implementamos a logica de ver se consta nas palavras lixo, se constar corta do bloco descrição. 
#também optimizamos o stage 2. Agora deve rodar bem melhor. A performance já ta certa segundo o gpt.
## Agora falta aplicar uma razão TF-IDF-like para servir como segundo filtro de palavras lixo,
# ex: se a gram aparece em 30% ou mais de todas as vagas. Deve ser palavra lixo ("job description", "we're hiring", etc).
# depois vou precisar usar o fuzzy matcher como forma de tratamento para os casos em que temos: "cross functional", "multi functional" etc.
#pq isso vai dar problema no power bi. (ex: Brazil e Brasil em linhas diferentes do gráfico)

SAVE_DIR = "C:\\Users\\ygorg\\OneDrive\\Documentos\\Pickles for remotive project"
os.makedirs(SAVE_DIR, exist_ok=True)

path = os.path.join(SAVE_DIR, "stage2_data.pkl")

with open (path, "wb") as d:
    pickle.dump ({
        "bloco_de_descricao": bloco_de_descricao,
        "negatives_to_machine_learning_stg3": negatives_to_machine_learning_stg3
    },d)

print ("stage 2 finished")

