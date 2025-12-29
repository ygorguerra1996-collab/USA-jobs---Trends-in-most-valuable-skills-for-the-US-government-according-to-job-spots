import requests #HTTP lib to interact with APIs
import sqlite3 #lib to handle data through databases.
import math
from dotenv import load_dotenv
import os
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import unicodedata #module that allows Unicode char manipulation 
import pandas as pd
import math
from spacy.lang.en.stop_words import STOP_WORDS #its gonna help in stage 2. to filter stopwords, not having to use doc.
import pickle

load_dotenv()
API_KEY = os.getenv ("API_KEY")
USER_EMAIL = os.getenv("USER_EMAIL")
##
if not API_KEY:
    raise RuntimeError ("API KEY NOT FOUND")

#Just explaining how the request to the API was made, Since the idea was to only work with IT roles.

url = "https://data.usajobs.gov/api/search"

headers = {
    
    "Host": "data.usajobs.gov",
    "User-Agent": USER_EMAIL,
    "Authorization-Key": API_KEY }

params = {
    "ResultsPerPage": 500,
    "Page": 1,
    "Series": "2210;1550;1560;0854;1520;1530;0801;1515;0343;0301;0080;0401;1301" 
} 


# "2210;1550;1560;0854;1520;1530": these are the clean series. But they get 85% of IT spots. to get closer to 
# 100% we will have to work with the other ones, that are ambiguous. These: 
#0801 – General Engineering
#1515 – Operations Research
#0343 – Management and Program Analysis
#0301 – Miscellaneous Administration
#0080 – Security Administration
#0401 – Natural Sciences
#1301 – General Physical Science

#just to know the good ones, since we will focus only in IT jobs:

#"2210",  # IT Management
#"1550",  # Computer Science
#"1560",  # Data Science
#"0854",  # Computer Engineering
#"1520",  # Mathematics
#"1530",  # Statistics
#"1515",  # Operations Research


response = requests.get()

#I ran the code with the requests (branch) in a different file, since it ran only once. Our logic
# was in this stage with the data that had been already stored in the db.

# From now on, we're simple working with the IT only jobs that were sent to the db in order to save requests.
# (USA jobs has a somewhat restrict requests limit)

conn = sqlite3.connect('USAjobs.db')
cursor = conn.cursor()

## as I already have all the jobs extracted in 10/12/2025 in the db
## in the table rolesUSAjobs. So to not waste requisitions,
## I changed the code to only work with whats already in the db.
## later I can change to do the role process automatized with airflow,
## but will make sure the pipeline is working propperly first.

cursor.execute('''
    select jobid, d_qualification_summary
               from rolesUSAjobs
            ''')

jobid_and_qualifications = cursor.fetchall() #returns a list of tuples

nlp = spacy.load("en_core_web_sm", disable= ["ner","parser","textcat"])

## begin of stuff imported from last try with remotive

bloco_de_descricao = [] #a list of dicts where each dict is a word from a job description. Its metadata-structured. Each word will bring its
# track, like the id, title, etc. Describing where it came from.
only_words_from_bloco = [] #will be used to store only description words from the each word dict above
dict_words_per_id = {}#will be used to prevent looking to words globally. Storying the words under de Job id
list_words_per_id= [] #will be used inside the dict_words_per_id to store the jd's words.
bigrams = []
trigrams = []
fourgrams = []
fivegrams = []

INVISIBLE_CHARS = [
    '\u200b', # zero-width space
    '\u200c', # zero-width non-joiner
    '\u200d', # zero-width joiner
    '\u202f', # narrow no-break space
    '\u2060', # word joiner
    '\uFEFF', # zero-width no-break space / BOM
]

separators = ['-', '_', '/', '.', '\n', '\r', '\t', ',', ';', ':']

def punctuation_removal(text):
    return ''.join( # the join function gathers the itens in a list. the first parameter for the join function is the segregator,
    #this join method is from the string object, not from the unicode lib. its formula is: separator_string(iterative_object)
    #the object can be a list, tuple etc
    #I'm saying that I don't want any character separating
    c for c in unicodedata.normalize('NFD',text) # c for c here is not the for loop! its a expressão geradora.
    #this c for c creates a generator. that is kind of a list but lighter. it doesnt stock in memory but provide the data on demand.
    #normalize is a function from the unicodedata lib that transforms the text into a 
    #unicode pattern. 'NFD' is the pattern that decomposes letters from punctuations in diferent chars.
    if unicodedata.category(c) not in ('Mn', 'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'So') #only joins the loop if its not a punctuation mark, or a exclamation mark, interrogation, etc. the function 'category' returns
    # a string that indicates if its a punctuation. 'mn' stands for Mark, non spacing (doesnt occupy segregated space), Pd is for dash and so on
     and unicodedata.category(c)[0] != 'C'
     and c not in INVISIBLE_CHARS) 

df = pd.read_excel ("C:\\Users\\ygorg\\Downloads\\Pasta Python codes\\USAjobs project\\ESCO skills - table_treated - Both ESCO and Skillner tables treated.xlsx")
df2 = pd.read_excel ("C:\\Users\\ygorg\\Downloads\\Pasta Python codes\\USAjobs project\\Teste impasse estagio 2.xlsx")
#this file gonna be used in stage 1
skills_set = set(df["skills_and_descriptions"]) #set removes duplicates, but doesnt guarantee same order, which, for my case, is irrelevant. 
#as I'm only going to have to check if the n-gram is in the excel file. "Set" makes the performance a lot better. Transforms the series object in 
#a set object

ngrams_to_be_cut = set (df2["word"]) #set with the most common ngrams that I could see that were not skill grams, but common words in every
#job description. Like "job description, hiring, hiring process" etc.

list_to_check=[]
## End of stuff imported from last try with remotive

jobids_only_list = [jobid for (jobid,qualifications) in jobid_and_qualifications] #transforming in separate lists
#to be able to use batch in nlp (it requires an iterable of texts)
qualifications_only_list = [qualifications for (jobid,qualifications) in jobid_and_qualifications]
# doing the same for the texts, now I can process all texts, tranforming in a doc, then attaching the
#jobid

qualifications_docked = nlp.pipe(qualifications_only_list,batch_size=200)

jobid_and_qualifications_docked = zip(jobids_only_list,qualifications_docked)

for (jobid,qualifications_doc) in jobid_and_qualifications_docked:
    stage1_listaftertreatment = [] #used to do the first treatment. A list that gathers all the descriptions, after normalizing, lematizing etc
    bigrams = [] #now I'm cleaning the lists regarding the ngrams, to prevent duplicates and unefficient memory use
    trigrams = []
    fourgrams = []
    fivegrams = []
    

## just to localize myself. I'm supposed to this:


     # a way to solve the perfoamnce problem,
    #but I'll have to gather the jobid with zip later.


    for token in qualifications_doc: #didn't used the listofwords to not have problems with the spacy later

        if token.pos_ in ('NOUN', 'VERB', 'ADJ'):
            word_lemma = token.lemma_
        else:
            word_lemma = token.text

        word_no_separator = word_lemma
        for sep in separators:
            word_no_separator = word_no_separator.replace(sep,' ')##o problema dos espaços que vi no stagio 2. Provavelmente
            #está aqui. Quando tem um separator, ele repoe por espaço. o problema é que esse espaço está indo junto pra
            #listaftertreatment.
        

        word_lower = word_no_separator.lower()
        word_stripped = word_lower.strip() #strip só remove espaços no inicio e no fim do texto.
        word_punctuation_removal = punctuation_removal(word_stripped)
        word_encoded_utf8 = word_punctuation_removal.encode ('utf-8',errors='ignore').decode('utf-8')


        word_encoded_utf8 = ' '.join(word_encoded_utf8.split()) #teoricamente isso aqui cuida dos espaços consecutivos caso precise.

        if word_encoded_utf8:#goes to the list only if its not a blank char.
            
            for w in word_encoded_utf8.split():

                stage1_listaftertreatment.append(w) #bingo. esta sendo zerado a cada loop.
        
    for i,word in enumerate(stage1_listaftertreatment): #enumerate creates a index for 
        #each word in the list. represented by the var "i" here.
        word_dict = {
        "jobid" : jobid, ## instead of gathering all of these other information in the bloco, I decided to go only with what matters.
##    "url" : url,
##    "title" : title,
##    "company_name" : company_name,
##    "category" : category,
##    "job_type" : job_type,
##    "publication_date" : publication_date,
##    "candidate_required_location" : candidate_required_location,
##    "salary" : salary,
        "word" : word,
        "index" : i
        }
        bloco_de_descricao.append(word_dict) #bloco de descrição contém um monte de dicts em que cada um é uma vaga.



for word_dict in bloco_de_descricao:
   
    if dict_words_per_id.get(word_dict["jobid"]) == None: 
        dict_words_per_id[word_dict["jobid"]]=[word_dict["word"]] #placing a list in this new key, thats going to be the first "word" in that id
            

    else:
        dict_words_per_id[word_dict["jobid"]].append(word_dict["word"]) # pasmem. Esse append não está sendo dado no Dict. Mas sim 



for jobid,words in dict_words_per_id.items(): #variable id is always the key and words, the words list
    for i in range (len(words)-1):
        bigrams.append ((jobid,[words [i],words[i+1]]))
    for i in range (len(words)-2):
        trigrams.append ((jobid,[words [i],words[i+1],words[i+2]]))
    for i in range (len(words)-3):
        fourgrams.append ((jobid,[words [i],words[i+1],words[i+2],words[i+3]]))
    for i in range (len(words)-4):
        fivegrams.append ((jobid,[words [i],words[i+1],words[i+2],words[i+3],words[i+4]]))

chunk_size = 1000 #the amount of n-grams that I want to send the db per commit, for performance
chunk = [] #going to be used for commiting the n-grams 1000 by 1000. to optimize performance
dict_jobid_index_to_remove_from_bloco = {}

for n_grams in [bigrams,trigrams,fourgrams,fivegrams]: #created a list with all the gram lists (bi,tri,four,five), where the "gram" is each of them
    for jobid,gram in n_grams: # since for example, each bi gram is a list of two words in the bigrams list, "gram" here is representing the two words
        clean_gram = " ".join (gram).lower()
        if clean_gram in skills_set: #theres a chance of this giving false positives. words like "and procedure", "and doing" can be in the esco excel
            chunk.append ((jobid,clean_gram,'Stage 1')) #appending to the queue list, to optimize performance by sending 1000 by 1000.

            indexes_to_remove = set() # set is an object similar to a list, but doesn't allow duplicates and doenst have indexes. i.e.: (4,5,6,7)
            # even if you try to put another 4 in this example, he will return (4,5,6,7). Its ideal to store unique elements. This one is
            #going to serve to store the indexes of the n-grams. Its necessary cause if we just do a if not word in n-gram. We can cut words
            # that are the same but are not present in the n-gram. So we have to create this set of indexes. So if the index is the same,
            # we can cut from bloco descricao. In our case set is useful cause we dont want duplicates

            bloco_vaga = [d for d in bloco_de_descricao if d["jobid"] == jobid] # a way to solve the problem with performance. Instead of
            #searching the whole bloco_de_descricao, we search only in the jobid that the ngram corresponds. Picks up only the dicts from that
            #specific jobid

            for i in range(len(bloco_vaga)-len (gram)+1): 
                slice_words = [d["word"] for d in bloco_vaga[i:i+len(gram)]] #I'm saying that I want to create a self-list based on the running
                ##through of the values of the key "word" in the dicts of the bloco_vaga. With the limit being the word we're on the loop plus, 
                #the length of the gram, so we dont break the list limit. THATS TO DIVIDE ALL THE WORDS INTO N-GRAMS WHOSE SIZE WILL VARY
                # ON THE GRAM SIZE, IN A NEW LIST, NAMED SLICE_WORDS
                # IMPORTANT!! But this list doesn't have all the words, its always subscribed at each loop. So if the gram is tri, its
                #going to have only 3 words, and the first word will have the same index as the one we're looking in the for i in range.
                if slice_words == gram: # I'm comparing if the words(only) of the specific jobid are present in the list of words underneath each
                # jobid stored in each grams list. I'm just searching for the positions of the founded ngram in the job description
 
                    indexes_to_remove.update([d["index"] for d in bloco_vaga[i:i+len(gram)]]) #the upgrade method is the same as 
                        #the append. But it wont break the logic because we're cleaning the set in the begining of this loop. So
                        #when the loop goes for the next gram, there's no acumulation because the set is cleaned for every new gram.
            # so, we created bi grams for every word in the bloco descricao in case the gram was a bi, trigrams if it was tri, and so on
            # also got the ids of these new grams. Using the vars slice_words and slice_ids
            # then if the words and ids matched with the ngram we got previously, we stored the indexes, for later removal in the bloco.

                    if dict_jobid_index_to_remove_from_bloco.get(jobid) == None:
                        dict_jobid_index_to_remove_from_bloco [jobid] = set(indexes_to_remove) #the idea is to create a dict that will contain in the keys,
            #the job ids and in the values the indexes of the words that need to be excluded from bloco_de_descricao. Had to place the 
            #"set" here too, to make a copy of the set indexes to remove and prevent data leak. Since if you change the value of a set
            #thats being referenced, it will impact in the other keys that its tied to.

                    else:
                        dict_jobid_index_to_remove_from_bloco[jobid].update(indexes_to_remove) 

            print(f" Stage1: n-gram found in esco-skillnet file:", clean_gram.strip())
            print (f" Stage1: jobid,Word queued for commit in the db:", jobid,clean_gram)
            if len (chunk) >= chunk_size: # if the chunk queue list reached 1000 or more itens, its time to commit to the db
                cursor.executemany ('''INSERT OR IGNORE INTO wordsindescription (jobid,word,stage_origin)
                VALUES (?,?,?)''', chunk)
                conn.commit()
                chunk = []

if chunk: # if theres still stuff in chunk, send the leftovers to the db
    cursor.executemany('''INSERT OR IGNORE INTO wordsindescription (jobid, word, stage_origin) VALUES (?, ?, ?)''', chunk)
    conn.commit()
    chunk = []

bloco_de_descricao [:] = [d for d in bloco_de_descricao if not (d["jobid"] in dict_jobid_index_to_remove_from_bloco and d["index"] 
in dict_jobid_index_to_remove_from_bloco[d["jobid"]])] #its working. Searches for the jobid from jobid from bloco_de_descricaoin the
#dict_jobid_index_to_remove... keys and if it finds it, searches for the indexes in the jobid values (set). so its searching for both 
#jobid and indexes. perfect for removing the index for each jobid, as described in the dict_jobid_index_to_remove... great.

#vars I'll use in stage2:
#bigrams, - list
#trigrams, - list
#fourgrams, - list
#fivegrams, - list
#ngrams to be cut, - dataframe
#bloco_de_descricao - list of dicts
#regarding the db, I'm gonna have to reopen in stage2, connect etc.

#now, saving the vars I'll use in stage 2 of the ETL in the pickle file, referencing in the stage 2 and keep working:

df_3 = pd.DataFrame(bloco_de_descricao[:500000])
df_3.to_excel("blocodescricaoposcorrecaodebug.xlsx",index= False)

print ("excel do bloco descricao para check gerado")

import os
import pickle

SAVE_DIR = "C:\\Users\\ygorg\\OneDrive\\Documentos\\Pickles for remotive project"
os.makedirs(SAVE_DIR, exist_ok=True)

path = os.path.join(SAVE_DIR, "stage1_data.pkl")

with open(path, "wb") as f: #creates a file with the name stage1_data.pkl...., wb is write binary. needed because pickle saves this way
    pickle.dump ({ #"dump" picks the object and saves in this file we're creating in the open. In this case, the object is a dict that
        #correlates the var names to their actual values
        "bigrams": bigrams,
        "trigrams": trigrams,
        "fourgrams": fourgrams,
        "fivegrams": fivegrams,
        "ngrams_to_be_cut": ngrams_to_be_cut,
        "bloco_de_descricao": bloco_de_descricao
    },f) #f is the alias to the file
print ("ok until here")

print ("stage 1 finished")

#maybe we still need some refactoration. Lets see if this runs tonight



















