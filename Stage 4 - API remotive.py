
## O que consideraremos como tech skill - foco do projeto:

##Linguagem
##Framework
##Ferramenta
##Tecnologia
##Protocolo
##Plataforma
##Infra / DevOps / Cloud
##Cyber

## só skills que tem curso sobre, ou que alguem falaria numa entrevista tecnica. nada de
## termos genericos como "administration" ou "data visualization tool", é coisa especifica.

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

WHITELIST_COMPOUND_TECH_SKILLS = { ## skills that were marked to be split but shouldn't be,
    #with the threshold for the c.similarity at 0,87. This list is to prevent spliting coumpound tech skills
    # this list was generated passing all skills that were marked to be split thanks to the cosine similarity vs threshold
    # to chat gpt...


    # AI / ML
    "machine learning",
    "artificial intelligence",
    "generative model",
    "ml model",
    "ml algorithm",

    # Security / Cyber
    "incident response",
    "threat intelligence",
    "intrusion detection",
    "intrusion detection prevention",
    "cybersecurity architecture",
    "cyber threat",
    "cyber incident",
    "cyber counter threat",

    # Networking / Protocols
    "internet protocol",
    "transport layer",
    "voice internet protocol",
    "lan wan",
    "lan wan internetwork",
    "tcp udp",

    # Cloud / Infra
    "cloud native",
    "hybrid cloud",
    "virtual machine",
    "virtual machines",
    "hyper v",
    "containerization",
    "cloud premise computing",

    # Cryptography
    "secure hash",
    "hash algorithm",
    "message digest",
    "message digest algorithm",
    "md5",
    "sha",
    "sha triple",

    # Systems / OS
    "windows server",
    "windows servers",
    "linux unix",

    # Data / Analytics
    "digital forensic",
    "forensic science",

    # Networking Hardware / Vendors
    "clearpass cisco",
    "cisco ios",
    "aruba switch",
    "router firewall",
    "lan switch",
    "san nas storage",

    # Dev / Architecture
    "cloud native modernization",
    "architecture topology protocol",
    "architecture installation integration",
    "architecture cloud",

    # General/ Validation with gpt on Cosine similarity between 0,80 and 0,87
    'agile framework',
    'cyber warfare',
    'generative model',
    'tape backup',
    'switch aruba',
    'cisco voip',
    'microsoft ms windows',
    'threat hunting',
    'radios sdr',
    'sprint planning',
    'performance tuning',
    'storage virtualization',
    'microsoft entra',
    'ms office',
    'geospatial modeling',
    'signal detection',
    'cns atm',
    'digital forensic',
    'predictive analysis',
    'switch cisco',
    'microsoft m365',
    'vulnerability scanning',
    'paas service',
    'hash algorithm sha',
    's3 knowledge',
    'algorithm sha',
    'model ml',
    'intune experience',
    'windows unix',
    'san nas storage',
    'microsoft azure',
    'iaas paas service',
    'firefox edge',
    'hub switch',
    'nessus technical',
    'framework rmf',
    'devsecops software',
    'md5 secure',
    'python powershell',
    'iam platform',
    'simulation modeling',
    'automation system',
    'automate infrastructure',
    'threat vulnerability risk',
    'internet protocol',
    'server desktop',
    'cloud native',
    'multiplexer concentrator',
    'integration voip',
    'framework dcwf',
    'vulnerability scan',
    'intelligence cyber',
    'accreditation network',
    'continuity operation',
    'vulnerability risk security',
    'availability authentication',
    'switch router',
    'intelligence ai ai',
    'lan level',
    'secure service',
    'forensic computer examiner',
    'cyber investigations',
    'protocol buffers',
    'generic routing encapsulation',
    'windows server',
    'windows servers',
    'malware forensic',
    'incident response',
    'threat hunting',
    'clearpass cisco',
    'windows server',
    'hyper-v',
    'hybrid cloud',
    'risk management framework (rmf)',
    'threat intelligence',
    'incident handling',
    'model-based systems engineering (mbse)',
    'sharepoint',
    'security+',
    'vulnerability assessment',
    'generative ai',
    'iacis certified forensic specialist',
    'verification and validation',
    'computer forensics',
    'devsecops',
    'kace systems management',
    'confidentiality integrity availability',
    'serverless applications',
    'cybersecurity architecture',
    'cyber warfare',
    'security content automation protocol',
    'web services',
    'secure software development',
    'data visualization software',
    'object-oriented programming',
    'voip',
    'data ingestion and transformation',
    'md5 hashing algorithm',
    'containerization microservices',
    'time series statistics',
    'it asset management'
}
whitelist_set = set(WHITELIST_COMPOUND_TECH_SKILLS)

BLACK_LIST_OF_GRAMS_ABOVE_COSINE_SIMILARITY_THAT_NEED_TO_BE_SPLIT = {
'sas spss',
'spss stata',
'voip vosip',
'splunk nessus',
'sas spss stata',
'tcp udp',
'tableau qlik',
'library itil',
'acas associate',
'jifm jfinsys jics',
'suite nessus',
'websense splunk',
'cisco av',
'qlik powerbi',
'aws ebs',
'pub sub',
'udp dds',
'av ip',
'o365 adobe',
'vulnerability encryption',
'clep ccaf dante',
'science physic',
'sysadmin infosec',
'probability statistic',
'static dynamic',
'language python',
'prof cert sec+',
'tcp ip'
}

blacklist_set = set(BLACK_LIST_OF_GRAMS_ABOVE_COSINE_SIMILARITY_THAT_NEED_TO_BE_SPLIT)
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



from sklearn.metrics.pairwise import cosine_similarity #imports the function that calculates cosine_simlarity
#it calculates how similar two vectors are in direction, not magnitude. result -1 for oposites, and 1 for identical
import numpy as np #numpy for numeric operations, very eficient

def should_split_compound(gram, gram_embeddings, threshold=0.8836): #function that says if the gram should be splitted or not
#gram is the string that I'll pass, model is the embeddings model (sentence-transformer)
# threshold = similarity limit that, if below this number, will tell that the grams don't make a whole concept
    tokens = gram.split() #treats and converts the gram in list

    if not (1 < len(tokens) <= 3): #guaranteeing unigrams wont be split
        return False,None # this line, is like the continue for loops but for functions. 
    # as we will have an outsider loop, when its false, its not going to do anything to the gram.
    # return in this case, CLOSES the function. 

    emb_full = gram_embeddings[gram].reshape(1,-1) #as we already calculated the embeddings for all grams
    #we're unpacking the embedding for the specific gram.
    
    emb_parts = model_embeddings.encode(tokens, convert_to_numpy=True)

    
    emb_mean = emb_parts.mean(axis=0, keepdims=True)
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

    # o keepdims é uma forma de manter no molde para poder passar para o modelo cosine_similarity.
    # até onde entendi, esse keepdims = true é uma forma do shape ficar (1,384) ao inves de (,384),
    # esse "1" é o numero no shape que sinaliza quantos embeddings estão ali. então esse keepdims
    # é a forma de forçar o array a manter uma dimensão extra que indica quantos embeddings existem ali.
   
    sim = cosine_similarity(emb_full, emb_mean)[0][0] # o Resultado desse cosine similarity vai sair algo como [[0.62]]. Então usamos [0] [0] para acessar o valor

    #cosine similarity is an object from sklearn.metrics.pairwise that compares how similar are two 
    # vectors on direction, not size. Result goes from -1 to 1. where 1= exactly the same, 0= no correlation
    # -1= opposite. It expects 2D matrixes. (thats why I had to use keepdims=True).

    #a formula para o cosine similarity é: 
    #   Produto escalar = (dim1 de A * Dim1 de B + dim2 de A * Dim2 de B....até o fim das 368 dimensões do embedding)
    # Norma A = Raiz quadrada de: (dim1A^2 + dim2A^2 + dim3A^2...)
    # Norma B = mesma coisa do A só que aplicado pra B

    #cosine similarity = produto escalar/norma A * norma B

    # Resultado próximo de 1 indica vetores na mesma direção = significados muito semelhantes, e 
    #com esse resultado vamos saber se precisa separar a gram ou não.

    return sim < threshold, sim #returns TRUE for spliting/ FALSE for nothing

    #keep the cosine at 0,88 and place the ones that are above it that should be split in a
    #black list
    #additionaly, check for all the splited with chat gpt to see the whitelist. -> first

    ##parei tentando fazer o gpt entender o que eu quero dele sobre as skills compostas... ele não tava
    ## entendendo, provavelmente por conta do tamanho das listas




unique_grams = list (set(grams for _,grams in positives_stage3_shortes_after_verb_removal)) #only grams,drops jobids
gram_embeddings = dict(zip(unique_grams,model_embeddings.encode(unique_grams,batch_size=512, convert_to_numpy=True))) #pegando só as grams e passando embedings
## A way that I found to pass all the grams through the sentence transformers once, instead of putting
## in a loop and decreasing performance
#so far we got gram + embedings in a dict

grams_to_split = []
map_grams_cosines = []

### isso aqui embaixo até então só ta gerando arquivos para auditar. ok to separando também o obj que vai cruzar com o positives_stage3_shortes_after_verb_removal
## para criar o novo que vai conter já tudo separadinho.
for gram,vector in gram_embeddings.items(): #iterating through gram+emb
    should_split,sim = should_split_compound(gram,gram_embeddings,threshold=0.8836)

    if gram in blacklist_set: 
        grams_to_split.append((gram,sim))

    elif should_split == True and gram not in whitelist_set:

        grams_to_split.append((gram, sim)) #placing the ones that are marked to be split in an object

    else: continue

    map_grams_cosines.append((gram,sim)) #only a map to check the cosines


df11 = pd.DataFrame(grams_to_split,columns= ["gram","cosine_similarity"])
df11.to_excel("grams_to_split.xlsx", index = False)

print ("check concluded")

df12 = pd.DataFrame(map_grams_cosines,columns= ["all_grams","cosine_similarity"])
df12.to_excel("allgrams_with_sims_to_check_cosines.xlsx",index = False) #to check why 
#some grams that were supposed to be separated, weren't.
print ("allgramscosinecheck made")

### FIM  da parte que ta gerando arquivos para auditar.

## OBS PARA AMANHA. COM THRESHOLD EM 0,87 AINDA TA FALTANDO QUEBRAR ALGUNS, APESAR DE TER QUEBRADO BASTANTE
## NÃO CHEQUEI O 0.87 SE CORTOU GRAMS QUE NÃO ERA PRA CORTAR. MAS VEMOS ISSO AMANHA, VOU TER QUE AUMENTAR UM POUCO
## DECIDED TO KEEP THE THRESHOLD AT 0.8836, with is kind of a high one. But to prevent separating skills that are compound, ("i.e. machine learning")
## I sent to gpt for validating the the ones that had from 0.8836 to 0.80 of cosine similarity. Having in mind that under 0.8 is almost
## certain that no coumpound skill is gonna be there.

## Good, compound skills between cosine similarity of 0,80 to 0,87 are in the whitelist now. I need to put the ones that are
## not compound skills that have C.S. higher than 0,87. in a black list and build logic to put them in the separation. NEXT STEP just increment in logic

#preciso pensar também no caso das trigrams... será que em todos os casos compensa quebrar em 3 palavras, ou tem alguns que vale manter 2 das tres juntas?
#aí encaixar a separação dos grams com o restante do código,

# depois decidir os casos de power bi, software power bi, powerbi (talvez esteja resolvido com o startswith que colocamos)


all_grams = []

###  positives_stage3_shortes_after_verb_removal para cruzar. e cruzar com o objeto que vai armazenar os que
# tem que splitar. é mais performatico.

positives_stage3_shorter_after_split_tied_skills = []
##putz, vai ter o problema com as 3 grams, pode ter casos que 2 são a mesma e tem uma colada. tenho que ver isso por fim.
set_grams_to_split = {gram for gram,sim in grams_to_split} #this is a set. its being created by a set comprehension

for jobid,grams in positives_stage3_shortes_after_verb_removal:
    if grams in set_grams_to_split:
        splited = grams.split()
        for words in splited:
            positives_stage3_shorter_after_split_tied_skills.append((jobid,words))
    else: positives_stage3_shorter_after_split_tied_skills.append((jobid,grams))


for jobid,grams in positives_stage3_shorter_after_split_tied_skills:
    all_grams.append(grams) # só as grams sem jobid

##all_grams é uma lista de listas, em que cada lista é uma gram

## pelo visto, vou implementar os filtros de "se o gram começa com verbo, experience with" etc,
# excluir, aqui . nesse objeto, all_grams


grams_as_strings = [g.strip() for g in all_grams if len(g.strip())>0] #just removing spaces
# and converting lists into strings

unique_grams = sorted(set(grams_as_strings)) #removing duplicates so clusterization is shorter

sample_size = 150000

first150k_unique_grams = unique_grams[:sample_size] #catching 150k of the unique grams (after deduplication) list
first150k_embeddings = model_embeddings.encode(first150k_unique_grams,batch_size=512, convert_to_numpy=True) #adding embeddings to these 150k

k =8

kmeans = KMeans(n_clusters=k,random_state=42,n_init='auto') 
kmeans.fit(first150k_embeddings) #passing these 150k to the kmeans for clusterization

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
