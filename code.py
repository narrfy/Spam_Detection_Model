"""TRAINING CODE"""
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

'''---load best model if saved---'''
try:
    with open('svm_model.pkl', 'rb') as file:#need to load bow that svm was trained on
        loaded_model = pickle.load(file)
        loaded_svm = loaded_model['model']
        loaded_bow = loaded_model['vectorizer']
except FileNotFoundError:
    print("Error: Model file not found.")
    loaded_svm = None
   
if loaded_svm is None:
    '''if no model saved, Run the test to find best model, includes loading training data, lem,vectorizing, word2vec, and test b/t modles''' 
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

    wnl = WordNetLemmatizer()
    word_lemmatized_string=[]

    #Lemmatize
    '''tokenize/'''
    token_documents = [[wnl.lemmatize(word) for word in word_tokenize(doc)]for doc in documents]
    word_lemmatized_string = [" ".join(doc) for doc in token_documents]
    print(word_lemmatized_string)
    print("WE ARE THROUGH LEMMATIZING THE DOCS")

    """load pre-trained embeddings"""
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
    '''average the word vectors in each document'''
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

    '''scale minmax scaler to make values 0 - 1'''
    glove_scaled_docs = MinMaxScaler().fit_transform(np.array(glove_avg))
    w2v_scaled_docs = MinMaxScaler().fit_transform(np.array(w2v_avg))
    fasttext_scaled_docs = MinMaxScaler().fit_transform(np.array(fasttext_avg))
    print("final_embeddings done")

    #Vectorize
    '''bow, tfidf, (different vectorizer models)'''
    bow = CountVectorizer()
    tfidf = TfidfVectorizer()

    tfidf_lem=tfidf.fit_transform(word_lemmatized_string)
    tfidf_norm = tfidf.fit_transform(documents)


    bow_lem=tfidf.fit_transform(word_lemmatized_string)
    bow_norm = bow.fit_transform(documents)


    '''###########################-Train-###########################################'''

    def save_results_to_file(results):
        with open("model_results.txt", "w") as file:
            for line in results:
                file.write(line + "\n")
    # Initialize a list to store results
    results = []           

    data_list = [glove_scaled_docs,w2v_scaled_docs,fasttext_scaled_docs,tfidf_lem,tfidf_norm,bow_lem,bow_norm]
    data_list_names = ["Glove_scaled_docs", "w2v_scaled_docs","fasttext_scaled_docs", "tfidf_lem", "tfidf_norm", "bow_lem", "bow_norm"]


    '''
    Best parameters for Decision Tree Classifier:
    {'criterion': 'gini', 'max_depth': 30}

    Best parameters for Support Vector Machine Classifier:
    {'C': 0.1, 'kernel': 'linear'}

    Best parameters for Random Forest Classifier:
    {'max_depth': 30, 'n_estimators': 100}
    '''

    nb = MultinomialNB()#NB
    dt = DecisionTreeClassifier(criterion= 'gini', max_depth= 30)#DT
    svm =SVC(C= 0.1, kernel= 'linear')#SVM
    rfc = RandomForestClassifier(max_depth= 30, n_estimators=100)  #RFC

    models = [nb,dt,svm,rfc]
    models_name = ["Naive Bayes","Decision Tree","SVM","Random Forest"]

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

    '''best is BOW_norm with SVM'''
    
    svm.fit(bow_norm,encoded)
    with open('svm_model.pkl','wb') as file:
        pickle.dump({'model':svm,'vectorizer':bow},file)
else:
    print("model loaded Successfully")


# Specify the data types for each column

#Test
testing = pd.read_csv('test.csv', dtype=np.unicode_)
testing.dropna(subset=['review_text'], inplace=True)
Steam_review = testing['review_text']

bow = loaded_bow
processed_test = bow.transform(Steam_review)

pred_label=loaded_svm.predict(processed_test)

testing['Pred_label']=pred_label

testing.to_csv('predicted_results.csv',index=False)




