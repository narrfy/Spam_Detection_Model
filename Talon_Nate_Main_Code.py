"""TRAINING CODE"""
'''May get error, press run to reload model'''
import pandas as pd
import numpy as np
import nltk
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import gensim.downloader
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score


'''
---load best model if saved---
:param a:model name
:param b: read bytes (allows any format -> helps with model)
:return: Saved model and vectorization from testing
'''
try:
    with open('svm_model.pkl', 'rb') as file:#need to load bow that svm was trained on
        loaded_model = pickle.load(file)
        loaded_svm = loaded_model['model']
        loaded_bow = loaded_model['vectorizer']
except FileNotFoundError:
    print("Error: Model file not found.")
    loaded_svm = None
'''
Find the best parameters for model and train on SMS
:return: File with best f1 macro score after anaylizing results
'''
if loaded_svm is None:
    '''Train to find th best model, includes loading training data, lematizing,vectorizing, word2vec, and test b/t models''' 
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
    # Encode labels
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(labels)
    
    #Preprocess
    #Lemmatize
    #return list of preprpocess and lemmatized words and sentences
    wnl = WordNetLemmatizer()
    word_lemmatized_string=[]
    token_documents = [[wnl.lemmatize(word) for word in word_tokenize(doc)]for doc in documents]
    word_lemmatized_string = [" ".join(doc) for doc in token_documents]
    print(word_lemmatized_string)
    print("WE ARE THROUGH LEMMATIZING THE DOCS")

    #load pre-trained word embeddings
    print("Loading glove gigaword")
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
    print("Loading google news")
    w2v_vectors = gensim.downloader.load('word2vec-google-news-300')#a vector with 300 features
    print("Loading fasttext subwords")
    fasttext_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
    
    print(glove_vectors)
    print(w2v_vectors)
    print(fasttext_vectors)
    
    print("embeddings loaded")
    
    #average the word vectors in each document to help with extremes
    #return list of words that were in word embedding with a vector of 100->300 features
    print('Averaging word vectors')
    
    #glove_avg = [np.mean([glove_vectors[word] for word in doc if word in glove_vectors], axis=0) for doc in token_documents]
    glove_avg = []
    for doc in token_documents:
        doc_embedding=[]
        for word in doc:
            try:
                doc_embedding.append(glove_vectors[word.lower()])#doing this to handle noun words that are kept upper case after lemmatizing
            except Exception as e:
                print(e)
        if doc_embedding:
            glove_avg.append(np.mean(doc_embedding,axis=0))
        else:
            glove_avg.append(np.zeros(100))
    
    #w2v_avg = [np.mean([w2v_vectors[word] for word in doc if word in w2v_vectors], axis=0) for doc in  documents]
    w2v_avg = []
    for doc in token_documents:
        doc_embedding=[]
        for word in doc:
            try:
                doc_embedding.append(w2v_vectors[word])#doing this to handle noun words that are kept upper case after lemmatizing
            except Exception as e:
                print(e)
        if doc_embedding:
            w2v_avg.append(np.mean(doc_embedding,axis=0))
        else:
            w2v_avg.append(np.zeros(300))
    
    #fasttext_avg = [np.mean([fasttext_vectors[word] for word in doc if word in fasttext_vectors], axis=0) for doc in  documents]
    fasttext_avg = []
    for doc in token_documents:
        doc_embedding=[]
        for word in doc:
            try:
                doc_embedding.append(fasttext_vectors[word.lower()])#doing this to handle noun words that are kept upper case after lemmatizing
            except Exception as e:
                print(e)
        if doc_embedding:
            fasttext_avg.append(np.mean(doc_embedding,axis=0))
        else:
            fasttext_avg.append(np.zeros(300))

    #scale minmax scaler to make values 0 - 1
    glove_scaled_docs = MinMaxScaler().fit_transform(np.array(glove_avg))
    w2v_scaled_docs = MinMaxScaler().fit_transform(np.array(w2v_avg))
    fasttext_scaled_docs = MinMaxScaler().fit_transform(np.array(fasttext_avg))
    print("Final embeddings done")

    #Vectorize with different methods
    bow = CountVectorizer()
    tfidf = TfidfVectorizer()
    
    '''tfidf'''
    tfidf_lem=tfidf.fit_transform(word_lemmatized_string)
    tfidf_norm = tfidf.fit_transform(documents)

    '''bow'''
    bow_lem=tfidf.fit_transform(word_lemmatized_string)
    bow_norm = bow.fit_transform(documents)
    print("Final vectorizations done")


    '''###########################-Train-###########################################'''
    #make file to save best model
    #Initialize a list to store results
    #return txt file from results of each test acc,f1micro,f1macro
    results = []
    def save_results_to_file(results):
        with open("model_results.txt", "w") as file:
            for line in results:
                file.write(line + "\n")
            
    #List of each preprocessessing text for input
    data_list = [glove_scaled_docs,w2v_scaled_docs,fasttext_scaled_docs,tfidf_lem,tfidf_norm,bow_lem,bow_norm]
    data_list_names = ["Glove_scaled_docs", "w2v_scaled_docs","fasttext_scaled_docs", "tfidf_lem", "tfidf_norm", "bow_lem", "bow_norm"]
    
    '''
    Results from GridSearch:
    Best parameters for Decision Tree Classifier:
    {'criterion': 'gini', 'max_depth': 30}

    Best parameters for Support Vector Machine Classifier:
    {'C': 0.1, 'kernel': 'linear'}

    Best parameters for Random Forest Classifier:
    {'max_depth': 30, 'n_estimators': 100}
    '''
    
    #Classification Models with best parameters listed above
    nb = MultinomialNB()#NB
    dt = DecisionTreeClassifier(criterion= 'gini', max_depth= 30)#DT
    svm =SVC(C= 0.1, kernel= 'linear')#SVM
    rfc = RandomForestClassifier(max_depth= 30, n_estimators=100)  #RFC

    models = [nb,dt,svm,rfc]
    models_name = ["Naive Bayes","Decision Tree","SVM","Random Forest"]
    
    #begin training
    #n_splits => # of cross validation splits
    #return results of each preprocess/embedding/vectorization teqhnique using different classification model
    print("Begin training")
    cf = KFold(n_splits = 10)
    for i, data in enumerate(data_list):
        result_str = f"Experiment: {data_list_names[i]}"
        
        results.append(result_str)
        
        for j, model in enumerate(models):
            acc_scores = []
            f1_micro_scores = []
            f1_macro_scores = []

            for k, (train_index, test_index) in enumerate(cf.split(data)):
                model.fit(data[train_index], encoded[train_index])
                y_pred = model.predict(data[test_index])
                acc = accuracy_score(encoded[test_index], y_pred)
                acc_scores.append(acc)
                f1_micro = f1_score(encoded[test_index], y_pred, average='micro')
                f1_micro_scores.append(f1_micro)
                f1_macro = f1_score(encoded[test_index], y_pred, average='macro')
                f1_macro_scores.append(f1_macro)
                #print(f"{k}/10")
            print("Experiment: ", data_list_names[i])
            print("Model: ", models_name[j])
            avg_acc = np.mean(acc_scores)
            print("Accuracy: ",avg_acc)
            avg_f1_micro = np.mean(f1_micro_scores)
            print("F1 Micro: ",avg_f1_micro)
            avg_f1_macro = np.mean(f1_macro_scores)
            print("F1 Macro:", avg_f1_macro)

            result_str = f"{models_name[j]} - Average accuracy: {round(avg_acc, 3)}, Average F1 micro: {round(avg_f1_micro, 3)}, Average F1 macro: {round(avg_f1_macro, 3)}"
            results.append(result_str)
            '''best is BOW_norm with SVM'''
        
    # Save the results to a text file
    
    save_results_to_file(results)
    loaded_bow = bow
    '''best is BOW_NORM with SVM, Has highest f1-macro score from results'''
    #save classification with pickle
    svm.fit(bow_norm,encoded)
    with open('svm_model.pkl','wb') as file:
        pickle.dump({'model':svm,'vectorizer':bow},file)
else:
    print("model loaded Successfully")
    
'''##### May get error, press run to reload model #####'''

#open test file, our Steam reviews documents
testing = pd.read_csv('test.csv', dtype=np.unicode_)
testing.dropna(subset=['review_text'], inplace=True)#return test file without NA values
Steam_review = testing['review_text']#the steam reviews

#use same preprocessing from best result of training
bow = loaded_bow #from saved model
processed_test = bow.transform(Steam_review)#preprocess like we did before
#predict label from reviews in steam document
pred_label=loaded_svm.predict(processed_test)#prediction
print("Predicted:", pred_label[:5])#print predictions 0 = ham 1 = spam

#return csv file with predicted result of test
filename = 'predicted_results.csv'
testing['Pred_label']=pred_label#add to testing dataset, label our data
testing.to_csv(filename,index=False)#make new csv with results
print("Saved file: ",filename)#print predictions 0 = ham 1 = spam




