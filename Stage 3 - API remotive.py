####----------------------Final stage, only one-word skills, after we extracted the ngrams and stored in the database. Just the leftovers----
#### no problem on removing stopwords here and not in the esco-skillnet, because the tokens left here are not related to any of these tables
#### I'm gonna deal only with the leftover words that didn't went to the db on stages 1 and 2.

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
import os


from sklearn.feature_extraction.text import TfidfVectorizer # for the machine learning later in this code
from sklearn.linear_model import LogisticRegression ## for the machine learning later in this code
from sklearn.model_selection import train_test_split ## for the machine learning later in this code
from sklearn.metrics import classification_report ## for the machine learning later in this code

nlp = spacy.load("en_core_web_sm",disable=["parser", "ner"])

conn = sqlite3.connect('USAjobs.db')
cursor = conn.cursor()

INVISIBLE_CHARS = [
    '\u200b', # zero-width space
    '\u200c', # zero-width non-joiner
    '\u200d', # zero-width joiner
    '\u202f', # narrow no-break space
    '\u2060', # word joiner
    '\uFEFF', # zero-width no-break space / BOM
]

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


LOAD_DIR = "C:\\Users\\ygorg\\OneDrive\\Documentos\\Pickles for remotive project"
path = os.path.join(LOAD_DIR, "stage2_data.pkl")


with open (path, "rb") as d:
    stage2_data = pickle.load(d)



bloco_de_descricao = stage2_data["bloco_de_descricao"]
negatives_to_machine_learning_stg3 = stage2_data["negatives_to_machine_learning_stg3"]

dict_words_per_id_stg3 = {}#needed to have a dict with the id being the key and all the words within a list as the value to be able to 
#work with the nlp

for word_dicts in bloco_de_descricao:
    if dict_words_per_id_stg3.get(word_dicts["jobid"]) is None:
        dict_words_per_id_stg3[word_dicts["jobid"]] = [word_dicts["word"]]

    else:
        dict_words_per_id_stg3[word_dicts["jobid"]].append(word_dicts["word"])


#words under each jobid, is going to be useful.
list_of_words = []
list_of_jobids = []



for jobid,words in dict_words_per_id_stg3.items():
    clean_words = ' '.join(words)
    list_of_jobids.append(jobid)
    list_of_words.append(clean_words) #need to create a list with only the job descriptions to pass to the nlp.pipe. As nlp.pipe provides the
    #advantage of processing in batch to optimize performance, it need to have a way to have acess to 1000 descriptions at one shot.

#list_of_words = [punctuation_removal(item) for item in list_of_words]


treated_doc = []
dict_jobid_word_to_remove_from_bloco_stage3 = {}

#this is the part I've just worked for upgrading the performance. Python doesnt work well with the doc object being stored/ used in an iteration.
#to create an object etc. Below is the problem solved.
#we solved it not working with the doc object, which is heavy, but only with the token's text (storing it in lists/ sets)
# the previous code, was just storing garbage tokens to send to machine learning as 0 and refreshing the bloco descricao. So the lines below
#already acomplish these points.

for jobid,doc in zip (list_of_jobids,nlp.pipe(list_of_words, batch_size=1000,n_process=1)): #n_process -1 utiliza todos os compartimentos do pc para acelerar o processo
    #mas da problema no windows
    words_to_remove = set()

    for token in doc:
        if (token.is_stop or 
            token.like_num or 
            token.is_currency or
            token.pos_ not in ('NOUN', 'PROPN', 'ADJ', 'VERB')):

            negatives_to_machine_learning_stg3.add(token.text.lower())
            words_to_remove.add(token.text.lower())
            if jobid not in dict_jobid_word_to_remove_from_bloco_stage3:
                dict_jobid_word_to_remove_from_bloco_stage3[jobid] = set ()
            dict_jobid_word_to_remove_from_bloco_stage3[jobid].add(token.text.lower())

bloco_de_descricao[:] = [d for d in bloco_de_descricao if not (d["jobid"] in dict_jobid_word_to_remove_from_bloco_stage3 and 
d["word"].lower() in dict_jobid_word_to_remove_from_bloco_stage3[d["jobid"]])] #removing from bloco_descricao jobids and words that are related
#to common words that are not skills.

print ("the size of the negatives passed to the model was:", len(negatives_to_machine_learning_stg3))

cursor.execute('''
select word from wordsindescription
where stage_origin <> 'stage_3'
''')

words_positive = cursor.fetchall() # gonna be used for negatives


X = []

Y = [] 

for (word,) in words_positive:
    X.append (word)
    Y.append (1)

for words in negatives_to_machine_learning_stg3:
    X.append (words)
    Y.append (0) 

x_train,x_test,y_train,y_test = train_test_split (X,Y,test_size=0.3,random_state=42)

vectorizer = TfidfVectorizer()

x_train_vec = vectorizer.fit_transform(x_train)

x_test_vec = vectorizer.transform(x_test)

model = LogisticRegression()

model.fit (x_train_vec,y_train)

preds = model.predict(x_test_vec)

print (classification_report(y_test,preds))

#modelo funcionou, 95% de acuracia. 

#Agora vou tentar resolver o problema de não poder passar a vaga inteira, se não ele siplesmente vai identificar que tem uma skill lá, sendo 
#completamente inutil pra mim.
#e também não posso passar grams curtas, pq se ele não tiver visto esse padrão antes, vai simplesmente atribuir zero aos vetores.
#eu passando a ngram um pouquinho maior, ele consegue ver pelo contexto aprendido que ali parece ter uma skill.

#por issso abaixo vou cortar em palavras maiores.

#needed to have a dict with the id being the key and all the words within a list as the value to be able to 
#need to recreate the dict_words_per_id_stg3 because only now the bloco descricao is cleaned from stopwords. As was passing
# it to the machine learning, it was returning all the stopwords.

dict_words_per_id_stg3_after_remov_stop = {} 

for word_dict in bloco_de_descricao:
    if dict_words_per_id_stg3_after_remov_stop.get(word_dict["jobid"]) is None:
        dict_words_per_id_stg3_after_remov_stop[word_dict["jobid"]] = [word_dict["word"]]

    else:
        dict_words_per_id_stg3_after_remov_stop[word_dict["jobid"]].append(word_dict["word"])


sixgrams_stage3 = []
sevengrams_stage3 = [] #separating in bigger grams so we can use the model not losing context
eightgrams_stage3 = []
ninegrams_stage3 = []
tengrams_stage3 = []

#If I pass the whole jobdescription to the model, its always going to return 1 and be completly useless to me. If I pass small grams, it doesnt 
#recognize the paterns (also exclude cases that are not in the training vocabulary)
#the solution is going to be pass bigger grams. So it can analyse the context and return only the parts where it probably contains a skill.
#later we're going to cut it even more as far as I understood.
#the strategy is going to be keep cutting until we have only the diamond.

for jobid,words in dict_words_per_id_stg3_after_remov_stop.items():

###    for i in range (len(words)-5):
###        sixgrams_stage3.append((jobid,[words[i],words[i+1],words[i+2],words[i+3],words[i+4],words[i+5]]))
###
###    for i in range (len(words)-6):
###        sevengrams_stage3.append((jobid,[words[i],words[i+1],words[i+2],words[i+3],words[i+4],words[i+5],words[i+6]]))
###
###    for i in range (len(words)-7):
###        eightgrams_stage3.append((jobid,[words[i],words[i+1],words[i+2],words[i+3],words[i+4],words[i+5],words[i+6],words[i+7]]))
###
###    for i in range (len(words)-8):
###        ninegrams_stage3.append((jobid,[words[i],words[i+1],words[i+2],words[i+3],words[i+4],words[i+5],words[i+6],words[i+7],words[i+8]]))

    for i in range (len(words)-9):
        tengrams_stage3.append((jobid,[words[i],words[i+1],words[i+2],words[i+3],words[i+4],words[i+5],words[i+6],words[i+7],words[i+8],words[i+9]]))


#aplicando o modelo as ngrams grandes, para ele identificar os trechos onde as grams estão, e irmos refinando.

all_big_grams_stage3 = sixgrams_stage3 + sevengrams_stage3 + eightgrams_stage3 + ninegrams_stage3 + tengrams_stage3

#the biggrams lists are with the words segreggated by commas, we wanna pass the role string, thats why were going to do the below:

texts = [" ".join(words) for _, words in all_big_grams_stage3]
jobids = [jobid for jobid,_ in all_big_grams_stage3]

for i in range (0,100):
    print (texts[i])

X_stage3 = vectorizer.transform(texts)

y_pred = model.predict(X_stage3)

positives_stage3 = [
(jobid,words)
for (jobid,words), pred in zip (all_big_grams_stage3,y_pred)
if pred ==1    
]

print (len(positives_stage3))

texts = []
jobids = []
all_big_grams_stage3 = []
sixgrams_stage3 = []
sevengrams_stage3 = []
eightgrams_stage3 = []
ninegrams_stage3 = []
tengrams_stage3 = []
gc.collect()

# segundo o gpt tudo faz sentido até aqui, agora precisamos pegar essas positivas e transformar em grams de 1-4
#o resultado já vai ser bem mais proximo das skills em si

# now I'm gonna transform the positive big grams in 1 to 4 grams. And apply the model on them.

def generate_ngrams(jobid,words,n_values=(1,2,3)):
    for n in n_values:
        for i in range (len(words)- n+1):
            yield(jobid,words[i:i+n]) #yield is a function that returns every grams not storing everything in memory. making it a lot faster
    #o yield é tipo um gerador que vai cuspindo os resultados um de cada vez ao invés de guardar na memoria. 
    #ele vai gerando uma por uma ao invez de guardar tudo numa variavel.
    #é uma função nativa do python. É como se fosse um return mas ao inves disso vai cuspindo cada resultado, depois parte pra proxima
    #iteração.

all_short_grams_positives = []
for jobid,text in positives_stage3:
    if isinstance (text,list):
        manywords = text
    else:
        manywords = text.split()

    all_short_grams_positives.extend(generate_ngrams(jobid,manywords)) # extend é a solução para quando você quer colocar os elementos de uma 
    #lista dentro de outra, sem criar uma lista dentro da lista. Ele só expande os valores, colocando tudo na mesma lista. Muito util.
    #é parecido com append. Só que inves de ficar [a,b,c[d,e,f]] ... fica [a,b,c,d,e,f]. É UMA FUNÇAO DO OBJETO LIST.

# esse loop ta fazendo o seguinte: Para cada gram no positive_stage3, ele chama a função, que vai gerar a unigram,
# cuspir pra all_short_grams, gerar a bigram, mesma coisa, trigram, mesma coisa e quadrigram. 
# Vai partir para a proxima gram do pisitive_stage3 e fazer o mesmo



texts = [" ".join(manywords) for _, manywords in all_short_grams_positives] 
jobids = [jobid for jobid, _ in all_short_grams_positives]

#applying batch to solve perfomance problem in the vectorizer part.
# as we have zillions of texts in the list "texts", trying to run all of them once crushes memory.So we divide in batches. 
# so we can process it in a lighter way.

#strategy 1 for performance gains:

batch_size = 5000
positives_stage3_shorter = [] 

for i in range (0,len(texts),batch_size): #its going to run through the texts list 5000 by 5000.the range function can receive 3 arguments
    #Beginning, end, step. With step being how many itens are going to be run through on each iteration.
    batch_texts = texts [i:i+batch_size] #cutting only the i (where we are in the loop) to the batch size; slicing the texts
    batch_jobids = jobids[i:i+batch_size] # same with the jobids
    batch_manywords = [all_short_grams_positives[j][1] for j in range(i,min(i+batch_size,len(all_short_grams_positives)))]
    ##batch_manywords = [all_short_grams_positives[j][1] for j in range (i,min(i+batch_size,len(all_short_grams_positives)))] #aqui, [j] representa o indice de cada
    #tupla, já que allshortgrams é uma lista de tuplas. e [1] significa que queremos pegar o segundo item de cada tupla que no caso são as grams.
    #note que ele não vai trazer também o primeiro item, que sera o jobid. Estamos querendo só pegar as grams aqui.
    #agora sobre o range, essa formula complexa é só pra garantir que não estoure a lista. Estou falando que o j vai ir de i (parte que estamos
    #iteração ali no for). até o que for menor: o i+o tamannho do batch, ou o tamanho da lista. isso garante que não vamos ultrapassar a lista
    X_batch = vectorizer.transform(batch_texts) #passei o batch_texts pq batch_manywords é uma lista de palavras, e precisa ser string.
    #mas mesmo assim precisei criar a batch manywords para conseguir identificar a qual gram pertence a previsão, depois.
    preds_batch = model.predict(X_batch) #aqui vamos receber do modelo o 0 e 1 que ele responde para cada gram.

    positives_stage3_shorter.extend([
        (jobid,manywords) #vamos criar a lista positives_stage3_shortes só com o jobid e a gram, zipando os jobids,manywords e respostas
        #da maquina onde for ==1. Pegando só o que contem skill.
        for jobid,manywords, pred in zip(batch_jobids,batch_manywords,preds_batch)
        if pred == 1
    ])




texts = []
jobids = []
all_short_grams_positives = []
gc.collect()


print ("we had this number of positive-short-grams returned:",len(positives_stage3_shorter))

SAVE_DIR = "C:\\Users\\ygorg\\OneDrive\\Documentos\\Pickles for remotive project"
os.makedirs(SAVE_DIR, exist_ok=True)

path = os.path.join(SAVE_DIR, "stage3_data.pkl")


with open (path,"wb") as d:
    pickle.dump({
        "bloco_de_descricao": bloco_de_descricao,
        "positives_stage3_shorter": positives_stage3_shorter,
        "vectorizer": vectorizer,
        "model": model
        },d
        )

#PAREI RODANDO UM TESTE DOS 100 POSITIVOS PARA MOSTRAR PRO GPT O QUE TEMOS ATÉ ENTÃO ANTES DE ELE CONFIRMAR SE VAMOS SEGUIR COM A ESTRATEGIA
# DOS EMBEDDINGS QUE ELE PROPOS

## problema detectado!!!! removi as stopwords do bloco depois de ter criado o dict words per id que é a fonte do que foi passado para ser
#analisado pelo machine learning.


##print (positives_stage3_shorter)

# o modelo ta com overfitting, nas biggrams já trouxe 3MM de positivos. Segundo o gpt, a maior chance é de como
#passamos 13k positivos, e 225 da lista fixa + os stopwords etc como 0. Passamos muito mais 1 do 0 e ele aprendeu que 1 é seguro.
#só estou esperando sair o resultado de quantos foram passados na list das stopwords para confirmar. Mas ou seja, tudo indica que até a 
#buxa é no modelo. Provavelmente vamos ter que equilibrar com mais negativas.

#averiguar se tinha que passar como lista, veio muito lixo ainda. Talvez o problema seja ter passado a string toda.

#provavelmente está estourando memoria ainda, provavelmente as listas que estou usando estão muito grandes.

##--------------------------- Machine Learning training-------------------------------

#OVERVIEW - (1) tf-IDF transforma palavras em numeros - (2) Logistic Regression aprende a separar skill de lixo - 
# - (3) train_test_split ajuda a validar o modelo
# - (4) classification_report mede o desemprenho e  (5) predict aplica o modelo a novos tokens

#### 1 - TF - IDF: ####################################################################

## tfidfvectorizer -- uma forma de você dar peso a cada palavra da sequencia, usando um vetor de numeros. E o 
## tfidfvectorizer faz isso. Dando peso para cada palavra dependendo da frequencia.

## o TF-IDF vem de Term frequency - Inverse Document Frequency. TF conta quantas vezes a palavra aparece em um documento
## enquanto o IDF é uma forma de balancear. já que ele penaliza palavras que aparecem em muitos documentos. (i.e. "the", "and")

# o tf conta a favor do ngram, enquanto o IDF ajuda a balancear, pq ve quantas vezes aquela palavra apareceu em todos os docs.

#é como se o tf iluminasse a lampada e o IDF enfraquecesse-a. Quanto maior.

# o vetor que tanto se fala, é a equação TF x IDF. O peso é o resultado dessa multiplicação. Indicando quão importante é aquela palavra
# em comparação com as outras no documento.

# o tf - idf são os valores que usamos para construir a vetorização . Ou seja, vai atribuir um numero para cada palavra. De acordo com sua
#Importancia. 

# Já que o ML não entende o texto cru, só numeros. Por isso que o TF - IDF é importante no contexto de machine learning

#Quando multiplicamos o tf pelo idf temos a representação numerica de cada palavra dentro de cada documento!!!!!!!!!!!!!!!!!!!!


## formula do TF: Nº de vezes que a palavra aparece no doc/ numero total de palavras no doc 

## formula do IDF: log (total de docs|1 + quantos doc contem a palavra)

####### 2 -  LogisticRegression  ################################################

# Modelo de classificação: Logistic Regression. É um classificador binário. no nosso exemplo vai decidir entre duas classes

#1 = skill, 0 = lixo.

# diferente do que o modelo sugere, não é só "regressão". no sentido de prever valores continuos. Não confundir com regressão linear.
#pq ai sim seria para prever valores continuos.
# ou seja você quer prever uma palavra como sendo skill x lixo , então usa o logistic regression.
# ele é usado quando quer prever uma classe discreta (binaria ou não binaria. No nosso caso é binaria.) Classe = Categoria.
# discreta = determinada, que você estipulou, não um numero qualquer. E binaria pq no nosso caso ele vai julgar 0 = lixo 1 = skill.
# por isso no nosso caso, Classe (skill x lixo) determinada (0 e 1) binaria (2 opções. Ou zero ou 1)
#ele usa função logistica (sigmoid) para transformar a previsão em probabilidade entre 0 e 1 e ai decide a classe.

#ele vai pegar cada palavra do texto e transformar em uma dimensão com o vetor (peso) como valor. 
# Se você tivesse 10.000 palavras, cada vetor estaria em um espaço de 10.000 dimensões.

#ele usa a função logistica sigmoid, para ver a probabilidade de ser skill da palavra.

# a formula é p(probabilidade) = 1/1 + e^-z

# onde: z = w1*x1 + w2*x2 + ... wn*xn + b                 # esse z é usado na formula acima

#x1, x2, x3... são os vetores tf - idf. Ou seja, a multiplicação do TF * o IDF de cada token.
#w1, w2, w3... são pesos passados pelo próprio modelo. Começam aleatórios, mas depois vão sendo ajustados a medida que o modelo avança.
# conforme você treina o modelo, ele vai aprendendo o peso de cada até ficar mais exato.

#TF-IDF transforma texto em números. A regressão logística aprende como usar esses números para decidir a classe. (no nosso caso, skill x lixo)

# Ou seja, o "W" é ajustado conforme passamos para o modelo (isso é skill, vs isso é lixo).
# O "W" é totalmente aprendido durante o treinamento.

#isso meio que faz sentido, já que a gente vira pra ele e fala "power bi" = 1 e "young" = 0. Você esta dando peso a aquelas palavras.

# beleza, agora falta aprender o que são o "e" e o "b" para entender a sigmoide (a primeira) e a de z (que é exatamente a regressão linear)

# sobre o "e" na formula, ele é um numero FIXO. sempre aprox. 2.71828. Nunca muda, que nem o pi. Mas relaxa, isso o python já sabe

# agora sobre o bias ou "b" da formula. Ele também é aprendido no treinamento. Quanto maior ele for, mais facil é o token ser classificado
#como "1". E quanto menor, mais dificil. Ele define a facilidade ou dificuldade que um token tem de ser classificado como 1.

#OLHA QUE LEGAL enquanto na regressão linear, você usa a formula y = x.w+b, a diferença pra logistica é que você usa a linear para transformar
# em uma probabilidade entre 0 e 1.

# Na logistica o que a gente faz é pegar a regressão linear e aplicar a sigmoide para ter a probabilidade. A sigmoide FORÇA o resultado 
# da regressão linear a encaixar entre um valor entre 0 a 1.



##### 3 - TRAIN_TEST_SPLIT

# Essa parte é só uma especie de compartamentalização. Voce fixa quantos % das grams (no caso vetores) vao para treino e quantas pra teste
#de acordo com o parametro test_Size. E o random_state fala se vai ser aleatorio ou sempre os mesmos;

# Essa parte serve para validar o modelo. Para termos certeza que ele realmente aprendeu com os dados ao inves de só replicar o que viu.

# isso é feito por uma função do skcit learn. a train_test_split:
# 
# se escreve no python assim:
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3, random_state = 42)

#A ordem das variaveis importa!!! pq se nao fica tudo bagunçado. ele retorna tudo na mesma ordem.

#X e y dentro da formula são objetos que estão armazenando os vetores das palavras (TF-IDF) que no caso é o X. e o y os labels, que são 
# as notas que você está dando (isso aqui é skill então é 1, o lixo é 0)

# a função retorna 4 resultados. Que no exemplo acima estamos armazenando nas variáveis. X_train, Xtest etc... 
# dentro dos parametros estamos passando X que é o resultado do TF-idf de cada palavra, y que são os labels para cada palavra.

# o random_state = 42 é só um parametro que garante que ele vai usar sempre as mesmas palavras como teste e treino.
# ou seja vai usar sempre as mesmas palavras como treino e as mesmas como teste

# o parametro test size=0.3 define que vamos usar 30% dos dados como teste, e o restante como treino. (70%)

# (X, y, test_size=0.3, random_state=42). Ela divide os dados emm conjuntos
# de treino e teste.

#IMPORTANTE, a variavel y_test que ele vai retornar, é o check que vamos fazer se ele aprendeu ou não.!!!!!!

# X_train e X_test vao retornar sempre a mesma coisa. Se você passar string ao inves dos vetores só pra fins didaticos, 
# ele vai retornar sempre a mesma coisa (os grams de treino e grams de teste), contando que você passar o random_state = 42

# se não passar random_state = 42 ele vai usar tokens aleatorios da sua lista.
# o y_train e o y_test também vao voltar igualzinho. O que você passou pra ele. contanto que você mantenha o random_state = 42.
# 
# o que muda conforme o modelo aprende é são as previsoes que você faz sobre X_test 
# É SEMPRE IMPORTANTE usar o train,test, split pq se você treinar ele com todos os dados. Ele pode simplesmente se lembrar das respostas
#e produzir a falsa sensação de que ele aprendeu. Mas quando vier dados novos ele vai errar tudo. Perigosissimo.

####### 4 - CLASSIFICATION REPORT - a validação do retorno obtido do modelo.

# model = LogisticRegression()
# model.fit(X_train_vec, y_train)

#Aqui, você vai passar a logisticRegression em uma variavel. E depois usar a função fit, passando o vetor das palavras, e as labels das palavras

# preds = model.predict(X_test_vec)

# essa variável preds vai conter o resultado das labels que ele previu (essa palavra predição confunde, mas o mais certo seria "advinhou")

# Ou seja, você vai ter o resultado do que ele aprendou. ele vai retornar o 0 e 1 de acordo com o que ele aprendeu sobre os vetores X.

# a validação vai ser comparar "preds" com "y_test
# 
# print (classification_report(y_test, preds))

# essa linha de codigo imediatamente acima é que vai ser o relatório de se o modelo acertou ou não.
# ele está batendo y_test que são os labels do test (gabarito) com o que veio de fato. É o batimento de fato.

#esse classification report retorna informações detalhadas do que houve: 

        # Precision - De tudo que o modelo disse que era aquela classe, quantos realmente eram?
        # Recall - De todos os exemplos reais daquela classe, quantos ele conseguiu identificar
        # f1 score - Média harmônica de precision e recall. Dá um balanço entre os dois.
        # support - Quantos exemplos reais dessa classe existem no teste.
        # 0 e 1	- Métricas por classe (no seu caso 0=lixo, 1=skill)
        # accuracy -% total de acertos em todas as classes
        # macro avg - média simples das métricas de todas as classes (não ponderada)
        # weighted avg -média ponderada pelo suporte de cada classe (mais fiel se classes estiverem desbalanceadas)


###########################  AGORA O CODIGO  PRA LER AMANHA E COMENTAR. TO LENDO E ENTENDENDO VELHO. ISSO É TUDO QUE VAMOS PRECISAR PRO PORTFOLIO
# 
# from sklearn.feature_extraction.text import TfidfVectorizer # Aqui estamos especificando que queremos só o submodulo feature... e . text
# significa que queremos só o sub-sub modulo de texto. e usamos import no tfidvectorizer pq essa é a classe. A logica do python é essa.
# navega pelos modulos com ponto, até chegar no nivel da classe. depois mais tarde instancia com () numa variavel


#from sklearn.linear_model import LogisticRegression # mesma coisa, importamos a logistic regression
#from sklearn.model_selection import train_test_split # mesma coisa que as anteriores
#
## Nosso dataset simples
#X = [
#    "I love this product",
#    "This is amazing and great",
#    "Absolutely fantastic experience",
#    "Terrible service and rude staff",
#    "I hate this item",
#    "Worst purchase ever"
#]
#
#y = [1, 1, 1, 0, 0, 0]  # 1 = positivo, 0 = negativo
#
## 1) Divide os dados em treino e teste # esse vai pegar os dados e compartamentizar. de acordo com os parametros que passamos, 30% (2 frases)
# serão usadas para teste, e o restante treino. Também estamos garantindo que usará sempre os mesmos como teste e treino. A função
# retorna sempre esses 4 objetos. por isso a ordem dessas variaveis é importante para não bagunçar. Até aqui estamos só separando os dados.
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.3, random_state=42
#)
#
## 2) Cria o vetorizador TF-IDF
#vectorizer = TfidfVectorizer() #instanciando o tfidfvectorizer que vai traduzir os tokens para vetores.
#
## 3) Aprende o vocabulário com os dados de treino e os transforma em vetores
#X_train_vec = vectorizer.fit_transform(X_train) # tranforma os dados de treino para vetores com a função .fit_transform. Aqui usamos fit_transforma
# que aprende o vocabulario (separa a frase em tokens separados, só com cada palavra unica) e transforma em vetores.
# o "fit" coloca o vocabulario em palavras diferentes, e o transform deixa em vetores.
#
## 4) Transforma os dados de teste usando o mesmo vocabulário 
#X_test_vec = vectorizer.transform(X_test) # mesma coisa com os de teste # note que aqui a função é diferente. usamos só transform. sem o fit_
# isso pq queremos que o modelo só atribua o tfidf as palavras que vieram do treino. As que estiverem no teste, mas não no treino,
# ele vai ignorar, atribuindo 0.0. Caso contrario ele vai 
# atribuir as palavras que só apareceram no teste no vocabulario, e pelo fato do teste ter menos palavras, logo vai distorcer os dados.
# pq as palavras unicas do teste vão ser mais raras (amostragem do teste é menor) logo vão ter IDF muito alto. o que vai abaixar
# os pesos (tf*idf) abaixando o peso que seria o justo das palavras de teste. e vai distorcer também o das palavras do treino, pq elas
# se tornaram menos raras.

# ponto importante: ele lembra dos vocabularios do x_train pq eles ficam armazenados no objeto vectorizer.
#
## 5) Cria e treina o modelo de regressão logística
#model = LogisticRegression() #instanciando a classe logisticregretion 
#model.fit(X_train_vec, y_train) # chamando a função. Aqui passamos os vetores do train e os labels do train para ele treinar.
#
## 6) Predições (o que o modelo acha que são os rótulos do teste)
#preds = model.predict(X_test_vec)  #pedimos para ele nos responder (a palavra predição é meio confusa, apesar que dá pra prever com isso)
# mas ta mais para "advinhar" ou nos responder.

#print (classification_report(y_test, preds)) #preds vai retornar as labels do test que passamos anteriormente. Aqui estamos
#peegando o gabarito (y test) e batendo com o que recebemos da maquina. esse print vai gerar o report mostrando o que deu certo x o que não.


### PROXIMOS PASSOS DO PROJETO:

## Eu consigo passar os n-grams + as unigrams (palavras que sobraram no bloco de descricao) pelo modelo, para ai ele classificar via
#logistic regression.
## O tfidf + os positivos e negativos que vou passar, vao ajudar o modelo a entender os padrões.
## Eu posso passar todos as palavras do bloco descrição em uma lista de unigrams (ou algo parecido) 
# e verificar AS UNI+NGRAMS pelo modelo.
# É acho que esse que vai ser a estrutura mesmo 




