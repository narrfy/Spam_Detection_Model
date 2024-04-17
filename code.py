import pandas as pd
import numpy as np
import nltk


from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
'''
import gensim.downloader
glove_model = gensim.downloader.load('glove-wiki-gigaword-100')
w2v_model = gensim.downloader.load('word2vec-google-news-300')
fasttext_model = gensim.downloader.load('fasttext-wiki-news-subwords-300')

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText
'''
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score


#Load Data (our training)
training = pd.read_csv("train.csv")
documents = training["v2"] # What we are training our model on -> convert to BoW vector
labels = training["v1"] # what we are trying to classify!

'''
#LOAD SPAM DATA (from class)
training = pd.read_csv("SPAM.csv")
documents = training["Message"] # What we are training our model on -> convert to BoW vector
labels = training["Category"] # what we are trying to classify!
'''

label_encoder = LabelEncoder()
encoded = label_encoder.fit_transform(labels)
#Preprocess
#snowball=SnowballStemmer(language="english")
word_lem = WordNetLemmatizer()
#stemmer = PorterStemmer()

#snowball_stemmed= []
word_lemmatized=[]
#stemmed = []

    #Lemmatize
for doc in documents:
    word_list = nltk.word_tokenize(doc)
    #stem_sentence = ' '.join([snowball.stem(w) for w in word_list])
    #snowball_stemmed.append(stem_sentence)
    lem_sentence = ' '.join([word_lem.lemmatize(w) for w in word_list])
    word_lemmatized.append(lem_sentence)
    #port_sentence = ' ' .join([snowball.stem(w) for w in word_list])
    #stemmed.append(port_sentence)
print("WE ARE THROUGH THE DOCS")

#Vectorize
#BoW
bow = CountVectorizer()

#bow_snowball = bow.fit_transform(snowball_stemmed)
bow_lem  = bow.fit_transform(word_lemmatized)
#bow_port = bow.fit_transform(stemmed)
bow_norm = bow.fit_transform(documents)
   

#Embed
#W2V
'''
print("Loading google news")
#w2v_model = gensim.downloader.load('word2vec-google-news-300')#a vector with 300 features
w2v_avg = [np.mean([w2v_model[word] for word in doc if word in w2v_model], axis=0) for doc in  X_hw3]

        #Glove
print("Loading glove gigaword")
#glove_model = gensim.downloader.load('glove-wiki-gigaword-100')
glove_avg = [np.mean([glove_model[word] for word in doc if word in glove_model], axis=0) for doc in X_hw3]

        #Fasttext
print("Loading fasttext subwords")
#fasttext_model = gensim.downloader.load('fasttext-wiki-news-subwords-300')
fasttext_avg = [np.mean([fasttext_model[word] for word in doc if word in fasttext_model], axis=0) for doc in  X_hw3]
'''
'''
print(glove_model)
print(w2v_model)
print(fasttext_model)
'''
print("embeddings loaded")


data_list = [bow_lem,bow_norm]
data_list_names = ["bow_lem", "bow_norm"]
#Train
nb = MultinomialNB()
dt = DecisionTreeClassifier()
svm =SVC()
rfc = RandomForestClassifier()
    #SVM
    #NB
    #DTC
    #RFC
cf = KFold(n_splits = 10)
for i,data in enumerate(data_list):
    
    acc_scores_nb = []
    f1_micro_scores_nb = []
    f1_macro_scores_nb = []
    
    acc_scores_dt = []
    f1_micro_scores_dt = []
    f1_macro_scores_dt = []
    
    
    acc_scores_svm = []
    f1_micro_scores_svm = []
    f1_macro_scores_svm = []
    
    acc_scores_rfc = []
    f1_micro_scores_rfc = []
    f1_macro_scores_rfc = []
    
    print("training and testing...")
    for j,(train_index, test_index) in enumerate(cf.split(data)):
        nb.fit(data[train_index], encoded[train_index])
        y_pred_nb = nb.predict(data[test_index])
        acc_nb = accuracy_score(encoded[test_index], y_pred_nb)
        acc_scores_nb.append(acc_nb)
        f1_micro_nb = f1_score(encoded[test_index], y_pred_nb, average='micro')
        f1_micro_scores_nb.append(f1_micro_nb)
        f1_macro_nb = f1_score(encoded[test_index], y_pred_nb, average='macro')
        f1_macro_scores_nb.append(f1_macro_nb)
        #print("nb done")
        
        dt.fit(data[train_index],encoded[train_index])
        y_pred_dt = dt.predict(data[test_index])
        acc_dt = accuracy_score(encoded[test_index], y_pred_dt)
        acc_scores_dt.append(acc_dt)
        f1_micro_dt = f1_score(encoded[test_index], y_pred_dt, average='micro')
        f1_micro_scores_dt.append(f1_micro_dt)
        f1_macro_dt = f1_score(encoded[test_index], y_pred_dt, average='macro')
        f1_macro_scores_dt.append(f1_macro_dt)
        #print("dt done")
        
        
        
        svm.fit(data[train_index],encoded[train_index])
        y_pred_svm = svm.predict(data[test_index])
        acc_svm = accuracy_score(encoded[test_index], y_pred_svm)
        acc_scores_svm.append(acc_svm)
        f1_micro_svm = f1_score(encoded[test_index], y_pred_svm, average='micro')
        f1_micro_scores_svm.append(f1_micro_svm)
        f1_macro_svm = f1_score(encoded[test_index], y_pred_svm, average='macro')
        f1_macro_scores_svm.append(f1_macro_svm)
        #print("svm done")
        
        
        rfc.fit(data[train_index],encoded[train_index])
        y_pred_rfc = rfc.predict(data[test_index])
        acc_rfc = accuracy_score(encoded[test_index], y_pred_rfc)
        acc_scores_rfc.append(acc_rfc)
        f1_micro_rfc = f1_score(encoded[test_index], y_pred_rfc, average='micro')
        f1_micro_scores_rfc.append(f1_micro_rfc)
        f1_macro_rfc = f1_score(encoded[test_index], y_pred_rfc, average='macro')
        f1_macro_scores_rfc.append(f1_macro_rfc)
        #print("rfc done")
        
        print(j+1,"/ 10")
    avg_acc_nb = np.mean(acc_scores_nb)
    avg_f1_micro_nb = np.mean(f1_micro_scores_nb)
    avg_f1_macro_nb = np.mean(f1_macro_scores_nb)
    
    avg_acc_dt = np.mean(acc_scores_dt)
    avg_f1_micro_dt = np.mean(f1_micro_scores_dt)
    avg_f1_macro_dt = np.mean(f1_macro_scores_dt)
    
    avg_acc_svm = np.mean(acc_scores_svm)
    avg_f1_micro_svm = np.mean(f1_micro_scores_svm)
    avg_f1_macro_svm = np.mean(f1_macro_scores_svm)
    
    avg_acc_rfc = np.mean(acc_scores_rfc)
    avg_f1_micro_rfc = np.mean(f1_micro_scores_rfc)
    avg_f1_macro_rfc = np.mean(f1_macro_scores_rfc)
    
    print("Experiment: ",data_list_names[i])
    print("Average accuracy np: ",round(avg_acc_nb,3))
    print("Average F1 micro np: ",round(avg_f1_micro_nb,3))
    print("Average F1 macro np: ",round(avg_f1_macro_nb,3))
    
    print("Average accuracy dt: ",round(avg_acc_dt,3))
    print("Average F1 micro dt: ",round(avg_f1_micro_dt,3))
    print("Average F1 macro dt: ",round(avg_f1_macro_dt,3))
    
    print("Average accuracy svm: ",round(avg_acc_svm,3))
    print("Average F1 micro svm: ",round(avg_f1_micro_svm,3))
    print("Average F1 macro svm: ",round(avg_f1_macro_svm,3))
    
    print("Average accuracy rfc: ",round(avg_acc_rfc,3))
    print("Average F1 micro rfc: ",round(avg_f1_micro_rfc,3))
    print("Average F1 macro rfc: ",round(avg_f1_macro_rfc,3))
'''   
#Test
testing = pd.read_csv('test.csv')
'''
    #Follow steps of model we decide on
