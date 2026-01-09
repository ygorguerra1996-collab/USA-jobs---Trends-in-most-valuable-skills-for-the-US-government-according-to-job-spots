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
import numpy as nd
from scipy import sparse
import gc


from sklearn.feature_extraction.text import TfidfVectorizer # for the machine learning later in this code
from sklearn.linear_model import LogisticRegression ## for the machine learning later in this code
from sklearn.model_selection import train_test_split ## for the machine learning later in this code
from sklearn.metrics import classification_report ## for the machine learning later in this code

from sentence_transformers import SentenceTransformer ##gonna be used for creating the embeddings (vector with 768 dims)

from sklearn.cluster import KMeans ## gonna be used for clustering
import os


#os 3 estagios estão muito bons, apesar de ter lixo ainda, mas tamos indo muito bem, até validei uma amostragem de 1k com o gpt
# conseguimos separar o lixo de "somos uma empresa assim assim..." e ficar com o que importa. Até o lixo aqui é proximo das skills.

start = time.time()
print ("starting to unpack pickle")

LOAD_DIR = "C:\\Users\\ygorg\\OneDrive\\Documentos\\Pickles for remotive project"
path = os.path.join(LOAD_DIR, "stage3_data.pkl")


with open (path, "rb") as d:
    stage3_data = pickle.load(d)


bloco_de_descricao = stage3_data["bloco_de_descricao"]
positives_stage3_shorter = stage3_data["positives_stage3_shorter"]
vectorizer = stage3_data["vectorizer"]
model = stage3_data["model"]     

end = time.time()

print (f"finished unpacking pickle, time needed: {end-start:.2f} seconds")


start = time.time()
print ("Starting to generate positive shorte grams excel check")

df5 = pd.DataFrame(positives_stage3_shorter[:1000000])
df5.to_excel("testing positive short grams.xlsx", index= False)

print ("excel para teste do positive short grams gerado")
end = time.time()
print (f"time needed to finish generating the excel for checking short grams: {end - start:.2f} seconds")

experience_qualifiers_to_cut = ["experience", "experience with", "experience in", "years of experience", "hands-on experience", "hands on experience", "practical experience", "professional experience", "relevant experience", "previous experience", "prior experience", "knowledge", "knowledge of", "working knowledge", "solid knowledge", "strong knowledge", "in-depth knowledge", "deep knowledge", "thorough knowledge", "general knowledge", "familiarity", "familiarity with", "basic familiarity", "strong familiarity", "proficiency", "proficiency in", "proficient in", "highly proficient", "advanced proficiency", "basic proficiency", "intermediate proficiency", "expert proficiency", "expertise", "expertise in", "deep expertise", "technical expertise", "subject matter expertise", "sme", "senior level experience", "lead level experience", "advanced experience", "ability", "ability to", "strong ability", "demonstrated ability", "proven ability", "capability", "capable of", "skill in", "skilled in", "hands-on", "hands on", "exposure to", "experience using", "experience working with", "worked with", "working with", "background in", "understanding of", "awareness of", "comfort with", "comfortable with", "years of", "minimum years", "x years", "minimum experience", "required experience", "preferred experience"'position','requirement','level','security','gs','application','technical','technique','development','datum','education','policy','engineering','qualification','grade','skill','technology','computer','employee','military','series','equivalent','problem','complex', 'year'

]

generic_non_tech_substantives_to_cut = ["activity", "activities", "mission", "missions", "task", "tasks", "responsibility", "responsibilities", "role", "roles", "function", "functions", "process", "processes", "procedure", "procedures", "operation", "operations", "workflow", "workflows", "work", "effort", "initiative", "initiatives", "objective", "objectives", "goal", "goals", "deliverable", "deliverables", "assignment", "assignments", "project", "projects", "program", "programs", "plan", "plans", "planning", "execution", "implementation", "support", "maintenance", "troubleshooting", "analysis", "evaluation", "assessment", "monitoring", "report", "reports", "documentation", "communication", "coordination", "collaboration", "interaction", "interface", "handling", "management", "administration", "operation support", "technical support", "user support", "customer support", "field support", "equipment", "hardware", "software", "systems", "system", "platform", "tools", "tool", "infrastructure", "environment", "resource", "resources", "materials", "components", "devices", "network", "networks", "data", "information", "content", "records", "files", "documents", "issues", "incidents", "problems", "requests", "tickets", "cases", "changes", "updates", "upgrades", "improvements", "enhancements", "solutions", "services", "service", "business", "organization", "company", "department", "team", "teams", "stakeholders", "clients", "customers", "users", "end users", "vendors", "partners", "requirements", "specifications", "standards", "policies", "guidelines", "procedural", "operational", "functional", "technical activities", "routine", "routines", "day-to-day", "daily activities", "general activities",'position','area','announcement','competency','method','solution','design','appropriate','concept','practice','principle','standard','research','base','training','duty','certification','employment','office','professional','current','resume','list','alternative','example','recommendation','eligible','appointment','federal','week','hour','leadership','study','degree','document','sufficient','addition','expert','non','dod','date','e','control','government','possess','applicant','general','minimum','responsible','advanced','category','complete','effective','enterprise','month','wide','defense','additional','order','mathematic','matter','air','acquisition','eligibility','major','physical','sector','material','applicable','job','quality','approach','membership','occupational','subject','necessary','army','civilian','graduate','action','public','condition','consideration','candidate','national','university','group','theory','vacancy','potential','relationship','accomplishment','background','institution','official','achievement','copy','instruction','technician','accordance','law','package','applicants','college','person','submit','troubleshoot','registration','relevant','decision','veteran','industry','low','maximum','title','description','career','accountability','enhancement','nebraska','oversight','permanent','act','oversee','successful','volunteer','commitment','large','budget','challenge','conscientious','occupation','american','broad','drug','impact','align','interpersonal','suitability','determination','competence','gpa','honor','improvement',

]

to_cut_prefixes = tuple(experience_qualifiers_to_cut + generic_non_tech_substantives_to_cut) # "startswith" doesnt accept lists, that why I'm converting them to tuples

# testei o positive stage3_shorter e ele tem 1-3 grams. tudo certo

#PROXIMOS PASSOS:

##📌 Etapas seguintes (bem simples):

##1) Gerar embeddings dos positivos
##
##Transformar cada token (unigram/bigram/trigram) em um vetor “que entende o significado”.
##→ Isso permite separar skills reais de palavras genéricas.
##
##Exemplo:
##
##“python”, “sql”, “docker” → ficam próximos.
##
##“create”, “maintain”, “learning” → ficam longe.
## O embedding gera um vetor que tem DIVERSAS dimensões. É como se ele tivesse colcoando cada palavra nesse espaço multidimensional de
##768 dimensões. Palavras semelhantes terão esses vetores próximos. enquanto muito diferentes, distantes. Assim ele mede a proximidade
## de significado.
## cada dimensão é como uma coordenada abstrata, que ele usa para "entender o significado" da palavra.
## O modelo já é treinado com milhões de frases da internet, livros etc. Ele sabe identificar inclusive palavras que tem mais de um 
## significado tipo "manga". Ele olha o contexto na frase. Ou seja é só aplicar o modelo na palavra ou frase para gerar os embeddings.
## pelo visto aqui vai ser o "sentence-transformers"
## Ou seja os embeddings também são vetores, mas é diferente do tf-idf. Enquanto o tfidf é calculado para medir o peso das palavras,
## com base em ocorrencia no documento*ocorrencia em todos os documentos. Em que um é o oposto do outro nessa equação; Os vetores do
## embeddings são calculados de acordo com o que o modelo já foi pré-treinado. Não existe “ocorrência no documento” nessa parte; o 
# modelo já sabe, pela sua experiência prévia, que “Python” e “SQL” são conceitos relacionados, mesmo que apareçam poucas vezes no seu dataset.

model_embeddings = SentenceTransformer('all-MiniLM-L6-v2') ## é o modelo 'all-MiniLM-L6-v2' que é leve, rápido e bom para tarefas de similaridade semântica
#ele que gera os embeddings de palavras. all-MiniLM é o modelo, "L6" significa que ele tem 6 camadas de atenção. E v2 é a versão dele.
#'all' indica que foi treinado para gerar embeddings para sentenças e palavras em geral nao só tarefas especificas.

##OBS: positives_stage3_shorter AQUI É O OBJETO QUE TEM O JOBID ATRELADO ATÉ ENTÃO

positives_stage3_shorter_after_listtreatment = []


for jobid,grams in positives_stage3_shorter:
    string_grams = ' '.join(grams)
    if string_grams.startswith (to_cut_prefixes): #ignoring the noise "experience with", "familiarity with" to get only tech skills 
        continue
    else:
        positives_stage3_shorter_after_listtreatment.append((jobid,string_grams))

batch_size = 500
positives_stage3_shortes_after_verb_removal = []
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


for i in range (0,len(positives_stage3_shorter_after_listtreatment),batch_size):
    batch = positives_stage3_shorter_after_listtreatment[i:i+batch_size]
    strings = [grams for jobid, grams in batch]

    for (jobid,grams),doc in zip(batch,nlp.pipe(strings, batch_size = batch_size)):
        if len(doc)> 0 and doc [0].pos_ == "VERB":
            continue

        positives_stage3_shortes_after_verb_removal.append((jobid,grams))


#add Multi-skill decomposition here then change var name below
# we're gonna attack first cases of "familiarity with python", "python experience", "python" etc.

## pelo que estava estudando, vamos usar embeddings em cada palavra dos grams que forem 2-3.
## caso o embedding deles seja similar, não separa, são correlacionados, tipo "machine learning"

## caso a similaridade for baixa, separa. que um não tem a ver com o outro. o GPT deu a ideia
## de usar um threshold de 0.75 para definir similaridade ou não. Amanhã vou retomar, mas vamos 
## partir desse código:

from sklearn.metrics.pairwise import cosine_similarity #imports the function that calculates cosine_simlarity
#it calculates how similar two vectors are in direction, not magnitude. result -1 for oposites, and 1 for identical
import numpy as np #numpy for numeric operations

def should_split_compound(gram, model, threshold=0.75): #function that says if the gram should be splitted or not
#gram is the string that I'll pass, model is the embeddings model (sentence-transformer)
# threshold = similarity limit that, if below this number, will tell that the grams don't make a whole concept
    tokens = gram.lower().split() #treats and converts the gram in list

    if not (1 < len(tokens) <= 3): #guaranteeing unigrams wont be split
        return False # this line, is like the continue for loops but for functions. 
    # as we will have an outsider loop, when its false, its not going to do anything to the gram.
    # return in this case, CLOSES the function. 

    emb_full = model.encode([gram]) # model refers to the SentenceTransformer
# we pass the entire gram as a single-element list because the model expects a list of texts
# the output is ONE embedding vector representing the semantic meaning of the full gram
    #emb_full represents the whole sentence meaning
    emb_parts = model.encode(tokens)#Model here is going to make reference to the sentence_transformers.
    # but now we are passing each word to the sentence transformers to have the 384D for each token.
    # emb_parts represents each word isolated meaning [embedding word1, embedding word2]

    #INSIGHT : we calculate both because we wanna se if the whole sentence meaning is similar to the
    # average meaning of the words? If yes, its a bi-tri gram. If not, its 2 different skills

    emb_mean = np.mean(emb_parts, axis=0, keepdims=True)
    #here, we are calculating a mean of the embeding parts, and the "axis=0" plays a key role:
    #IMPORTANT: the mean is made by calculating, for example in a bigram:
        #(dim1token1 + dim1token2)/2 ... (dim2token1 + dim2token2)/2...
        #he calculates the means between the dimensions and leaves on a list.machine   → [10, 20, 30] Ex hipotetico se fossem só 3 dim:
                                                                            #learning  → [40, 50, 60
                                                                            #mean     → [25, 35, 45]
    #Axis=0 calculates the mean VERTICALLY. (between dimensions)...Which is adequate for our case. 
    # As we want to calculate mean between all the dimensions of the vector
    # if it was Axis=0 or if we didn't mention it in the parameters, it would calculate the mean
    # from all dimensions of token1 and the mean from all dimensions from token2. So we would have
    # only 2 numbers instead of 384. For our case, its important to have the 384 means.

    ##VER O KEEPDIMS AMANHÃ. TO CANSADO


    sim = cosine_similarity(emb_full, emb_mean)[0][0]

    return sim < threshold




all_grams = []

for jobid,grams in positives_stage3_shortes_after_verb_removal:
    all_grams.append(grams)

##all_grams é uma lista de listas, em que cada lista é uma gram

## pelo visto, vou implementar os filtros de "se o gram começa com verbo, experience with" etc,
# excluir, aqui . nesse objeto, all_grams


grams_as_strings = [g.strip() for g in all_grams if len(g.strip())>0] #just removing spaces
# and converting lists into strings

unique_grams = sorted(set(grams_as_strings)) #removing duplicates so clusterization is shorter

sample_size = 150000

first150k_unique_grams = unique_grams[:sample_size]
first150k_embeddings = model_embeddings.encode(first150k_unique_grams,batch_size=512, convert_to_numpy=True)

k =8

kmeans = KMeans(n_clusters=k,random_state=42,n_init='auto')
kmeans.fit(first150k_embeddings)

cluster_dict = {}

first150k_labels = kmeans.predict(first150k_embeddings)
for gram, cluster_id in zip (first150k_unique_grams,first150k_labels):
    cluster_dict[gram] = int(cluster_id)

batch_size = 500
remaining_grams = unique_grams[sample_size:]

for i in range (0,len(remaining_grams),batch_size):
    batch = remaining_grams[i:i+batch_size]

    batch_embeddings = model_embeddings.encode(batch,batch_size=512,convert_to_numpy=True)

    batch_labels = kmeans.predict(batch_embeddings)

    for gram, cluster_id in zip(batch,batch_labels):
        cluster_dict[gram] = int(cluster_id)

print (cluster_dict)

clusterdict_inlist_to_slice_then_dictagain = dict(list(cluster_dict.items())[:500000])

df10 = pd.DataFrame([{"gram": gram, "cluster_id": cluster_id} for gram,cluster_id in 
clusterdict_inlist_to_slice_then_dictagain.items()])
df10.to_excel("clusterIDtocheck.xlsx",index=False)


print ("cluster dict concluído")


#important!!:
#I used the clusterdict in excel to check for words that were poluting the embeddings. Generating 
#disordered clusters.
# I got the most frequent intial words on the grams them checked if the top 20 were tech-related skills
#if they weren't, I added to the lists "experience_qualifiers_to_cut" and " experience_qualifiers_to_cut"
# in order to filter the noise.
# 
 
##boa, cluster 3 ta ficando com cara de techskill. Agora preciso decidir depois se vou pra tratar
## os dados que tem skills misturadas tipo "html sql", "python java" etc.




##Ordem de resolução:
##1- Separar skill normal, de tech skill. vamos focar nas tech skills - tecnologia, ferramenta, linguagem, framework ou plataforma.
    #pra isso:
    #excluir casos das listas fixas.
    #excluir dos 1-3 grams as grams que começam com verbo.
        #isso não da pau pq eu vou retirar a gram que é "experience with python" e ficar com a "python"
    # remover grams que começam com paalvras que obviamente não são tech skills, como "experience with"
    # position #hiring etc... - ok. Removi tudo que começava com isso sem medo de perder skill pq,
    #pela logica dos grams, na outra ponta se tem experience with python, vai ter só "python". Já que eu
    #gerei 1-3 grams no texto todo.



##2 - resolver o problema de 2,3 skills na mesma gram. Vamos partir pra isso antes de normalização
##3- Resolver o problema de experience with python, python, familiarity with python
##4 - Decidir o que fazer com casos ambiguos como "program" etc
