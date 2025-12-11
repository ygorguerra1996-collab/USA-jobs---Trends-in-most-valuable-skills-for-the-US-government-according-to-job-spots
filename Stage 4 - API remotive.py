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


LOAD_DIR = "C:\\Users\\ygorg\\OneDrive\\Documentos\\Pickles for remotive project"
path = os.path.join(LOAD_DIR, "stage3_data.pkl")


with open (path, "rb") as d:
    stage3_data = pickle.load(d)


bloco_de_descricao = stage3_data["bloco_de_descricao"]
positives_stage3_shorter = stage3_data["positives_stage3_shorter"]
vectorizer = stage3_data["vectorizer"]
model = stage3_data["model"]     


df5 = pd.DataFrame(positives_stage3_shorter[:1000000])
df5.to_excel("testing positive short grams.xlsx", index= False)

print ("excel para teste do positive short grams gerado")

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

all_grams = []

for jobid,grams in positives_stage3_shorter:
    all_grams.append(grams)

##all_grams é uma lista de listas, em que cada lista é uma gram

grams_as_strings = [" ".join (g).strip() for g in all_grams if len(" ".join(g).strip())>0]

unique_grams = sorted(set(grams_as_strings)) #retuns 557k unique grams
#####temos um problema aqui. A minha lista de unique_grams tem 577k de registros. E segundo o GPT
#####é muito pesado gerar os embeddings para tudo isso. O problema não é nem os clusters, e sim
##### os embeddings.
#### vou ter que dar um jeito de filtrar.

#### ESTRATEGIA possivel, embedar 150k e treinar o kmeans com esses 150k e depois embedar o restante
#### em batches e passar pelo k-means. Segundo o GPT isso vai ser leve.

sample_size = 150000

first150k_unique_grams = unique_grams[:sample_size]
first150k_embeddings = model_embeddings.encode(first150k_unique_grams,batch_size=512, convert_to_numpy=True)

k =20

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

###onde estou agora: decidir se vamos pegar aquela parte de actividades que descreve o que a pessoa
### vai fazer no dia a dia também, ou só pegar os requirements/ skills/ must have.

### não daria muito trabalho provavelmente, talvez fosse só uma questão de no stage 1
### cortar tudo que vem antes de "requirements", "skills", "what we're looking for"...

## o GPT me aconselhou a pegar só as skills e excluir as atividades. Pq se não o gráfico vai 
## virar uma zona. To inclinado a isso também.

## outro ponto é, escolher os clusters depois. talvez o melhor seja dar um passo atras e só trabalhar
## com o texto que vem antes desses pontos. O que vai ajudar demais na hora incluso de escolher os 
## clusters.

## ---------------------------------------------------------------------------------------------------


## A ESTRATÉGIA PARA NÃO DAR PAU NA HORA DE CORTAR DE positives_stage3_shorter (PARA MANTERMOS O JOB ID) vai ser, depois do clustering,
##usar um dicionario, para a operação ser 0(1). Então depois do clustering vamos fazer algo assim: cluster_dict = {gram: cluster_id,....}
#e bater esse dict com a positives_stage3_shorter


##embeddings = model.encode(unique_grams) ##passando as 1-3 grams que sobraram na ML do estagio anterior para gerar os embeddings.
#a função encode pega cada string e transforma em um vetor de numeros



##k = 20

## explicação sobre os clusters que o kmeans vai fazer.
## clustering é olhar para as 768 ou 384 dimensoes que foram atribuidas como vetores a cada gram, e vai AGRUPAR POR SEMELHANÇA.
## de acordo com o que essas dimensões mostram de cada palavra. como tem um MONTE, ele consegue olhar e entender o significado dessas palavras
## de acordo com a proximidade entre essas dimensões.
## O kmeans vai as grams e atribuir um cluster para cada gram. É como se tivesse uma multidão de pessoas, e ele falasse "idosos na fila 1"
## "crianças na fila 2", "programadores fila 3"... etc. está agrupando por semelhança;


##kmeans = KMeans(n_clusters=k,random_state=42, n_init='auto') 
##
##labels = kmeans.fit_predict(embeddings)
##
##cluster_dict = {gram: int(cluster_id) for gram, cluster_id in zip(unique_grams, labels)}
##
##print (cluster_dict)

#### REALMENTE, HA ALGUMAS PALAVRAS QUE ESTÃO VINDO JÁ COLADAS (TALVEZ DO PROPRIO JSON OU DO TRATAMENTO DO BS4.)
### E SÃO POUCAS, UMAS 5K DOS 1MM DO BLOCO DESCRICAO. ACONTECE QUE POR SUA RARIDADE O TFIDF DELA DA ALTO E ELA PASSA COMO POSITIVA PELO ML.
### E O CLUSTER TAMBÉM COSTUMA ESCOLHER ESSAS PALAVRAS PARA EMCABEÇAR A FILA. POR ISSO ESTAMOS VENDO ELAS MUITO NOS CLUSTERS.
### AGORA TEM QUE VER SE NÃO É VIAJEM DO GPT ISSO.


##obs tem umas grams que vieram com palavras coladas, provavelmente na hora de separar os /z etc, não colocarmos espaço.
## o gpt falou que mesmo assim ele identifica, mas precisamos averiguar isso.









##2) Fazer clustering nos embeddings
##
##Juntar vetores parecidos.
##Use HDBSCAN (melhor) ou K-means (mais simples).
##
##Isso vai formar grupos como:
##
##Cluster 1 → python / sql / tableau / power bi
##
##Cluster 2 → create / maintain / learning / assessment
##
##Cluster 3 → articular / captivate / rise
##
##3) Identificar quais clusters representam SKILLS
##
##Você olha apenas os top tokens de cada cluster.
##Se for skill → você marca como “skill cluster”.
##Se for verbo genérico → ignora.
##
##É rápido e totalmente manual, mas é só olhar 20 clusters, não 6 milhões de tokens.
##
##4) Gerar seu dicionário final de skills
##
##Pegue todos tokens nos clusters marcados como SKILLS.
##Pronto: você criou seu dicionário final.
##
##5) Filtrar os positivos
##
##Agora é fácil:
##Só manter tokens cujo cluster = “skill”.
##
##Isso reduz de milhões para só as skills reais.
