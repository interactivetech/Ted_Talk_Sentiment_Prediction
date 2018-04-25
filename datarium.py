
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import textacy
import gensim
import nltk

import re

from sklearn.preprocessing import StandardScaler

from gensim.models.doc2vec import TaggedDocument

from nltk.tokenize import sent_tokenize


# In[ ]:


def get_years():
    def get_num(s):
        ls = re.findall(r'-?\d+\.?\d*', s)
        if len(ls)==0:
            years.append(years[-1])
        else:
            try:
                if int(ls[0]) < 100:
                    years.append(years[-1])
                else:
                    years.append(int(ls[0]))
            except:
                years.append(years[-1])

    years = []
    df['event'].apply(get_num)

    scaler = StandardScaler()
    scaler.fit(np.array(years).reshape(-1, 1))

    years_scaled = scaler.transform(np.array(years).reshape(-1, 1))

    return years_scaled


# In[2]:


def split_data(test_frac=0.1, undersample=False):
    def get_undersampled(df, threshold=200, window=10):
        indices = df['label'].value_counts()[df['label'].value_counts() > threshold].index.tolist()

        for ind in indices:
            over_df = df[df['label'] == ind].reset_index(drop=True)
            df = df.drop(df[df['label'] == ind].index)
            to_drop = np.random.randint(threshold-window, 
                                        threshold+window)

            trans_ids = np.random.choice(range(len(over_df)),
                                         to_drop)

            dfs_to_add = over_df.iloc[trans_ids]
            df = pd.concat([df, dfs_to_add]).reset_index(drop=True)


        # Shuffle result
        df = df.sample(frac=1).reset_index(drop=True)

        return df

    df = pd.read_csv("data/Merged_dataset.csv")
    if undersample:
        df = get_undersampled(df, threshold=200, window=10)
    
    df_test = df.sample(frac=test_frac)
    df_train = df.drop(df_test.index)
    
    df_train.to_csv("data/train_raw.csv", index=False)
    df_test.to_csv("data/test_raw.csv", index=False)


# In[3]:


def get_oversampled_train(threshold=50, window=10):
    df_train = pd.read_csv("data/train_raw.csv")

    indices = df_train['label'].value_counts()[df_train['label'].value_counts() < threshold].index.tolist()
    
    new_df_train = df_train
    for ind in indices:
        poor_df = df_train[df_train['label'] == ind].reset_index(drop=True)
        to_add = np.random.randint(threshold-window, 
                                   threshold+window) - len(poor_df)
        trans_ids = np.random.choice(range(len(poor_df)), 
                                     to_add, 
                                     replace=True)
        dfs_to_add = poor_df.iloc[trans_ids]
        new_df_train = pd.concat([new_df_train, dfs_to_add]).reset_index(drop=True)
        
    # Shuffle result
    new_df_train = new_df_train.sample(frac=1).reset_index(drop=True)
    
    return new_df_train


# In[4]:


def create_bow(oversample=False, description=False):
    print("Reading the data...")
    
    if oversample:
        df_train = get_oversampled_train()
    else:
        df_train = pd.read_csv("data/train_raw.csv")
    
    df_test = pd.read_csv("data/test_raw.csv")
    
    print("Creating the corpus...")
    corpus_train = textacy.Corpus(lang='en', texts=df_train['transcript'].tolist())
    corpus_test = textacy.Corpus(lang='en', texts=df_test['transcript'].tolist())
    
    tokenized_docs_train = (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) for doc in corpus_train)
    tokenized_docs_test = (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) for doc in corpus_test)
    
    print("Generating BOW...")
    vectorizer = textacy.Vectorizer(apply_idf=False, min_df=4, max_df=0.95)
    bow_train = vectorizer.fit_transform(tokenized_docs_train)
    bow_test = vectorizer.transform(tokenized_docs_test)
    
    bow_train = pd.DataFrame(bow_train.toarray())
    bow_test = pd.DataFrame(bow_test.toarray())
    
    if description:
        pd.concat([bow_train, df_train['label']], axis=1).to_csv("data/bow_train_description.csv", index=False)
        pd.concat([bow_test, df_test['label']], axis=1).to_csv("data/bow_test_description.csv", index=False)
    else:
        pd.concat([bow_train, df_train['label']], axis=1).to_csv("data/bow_train.csv", index=False)
        pd.concat([bow_test, df_test['label']], axis=1).to_csv("data/bow_test.csv", index=False)


# In[11]:


def create_tfidf(oversample=False, description=False):
    print("Reading the data...")
    
    if oversample:
        df_train = get_oversampled_train()
    else:
        df_train = pd.read_csv("data/train_raw.csv")
        
    df_test = pd.read_csv("data/test_raw.csv")
    
    print("Creating the corpus...")
    corpus_train = textacy.Corpus(lang='en', texts=df_train['description'].tolist())
    corpus_test = textacy.Corpus(lang='en', texts=df_test['description'].tolist())
    
    tokenized_docs_train = (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) for doc in corpus_train)
    tokenized_docs_test = (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) for doc in corpus_test)
    
    print("Generating TF-IDF...")
    vectorizer = textacy.Vectorizer(apply_idf=True, norm="l2", min_df=4, max_df=.95)
    tfidf_train = vectorizer.fit_transform(tokenized_docs_train)
    tfidf_test = vectorizer.transform(tokenized_docs_test)
    
    tfidf_train = pd.DataFrame(tfidf_train.toarray())
    tfidf_test = pd.DataFrame(tfidf_test.toarray())
    
    if description:
        pd.concat([tfidf_train, df_train['label']], axis=1).to_csv("data/tfidf_train_description.csv", index=False)
        pd.concat([tfidf_test, df_test['label']], axis=1).to_csv("data/tfidf_test_description.csv", index=False)
    else:
        pd.concat([tfidf_train, df_train['label']], axis=1).to_csv("data/tfidf_train.csv", index=False)
        pd.concat([tfidf_test, df_test['label']], axis=1).to_csv("data/tfidf_test.csv", index=False)


# ### Doc2Vec

# In[6]:


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield TaggedDocument(doc, [self.labels_list[idx]])
                
# Parentheses describe the speech in action
def remove_parentheses(s):
    return re.sub("[\(\[].*?[\)\]]", "", s)

def get_vector(d2v_model, text):
    d2v_model.random = np.random.RandomState(42)
    return d2v_model.infer_vector(text)


# In[7]:


def create_doctovec(train=False, num_epochs=10, oversample=False, description=False):
    print("Reading the data...")
    
    if oversample:
        df_train = get_oversampled_train()
    else:
        df_train = pd.read_csv("data/train_raw.csv")
        
    df_test = pd.read_csv("data/test_raw.csv")
    
    text_train = df_train['transcript']
    names_train = df_train['name']
    text_test = df_test['transcript']
    names_test = df_test['name']
    
    train_data = [nltk.word_tokenize(d.lower()) for d in text_train]
    test_data = [nltk.word_tokenize(d.lower()) for d in text_test]
    
    if train:
        print("Doc2Vec Training...")
        #iterator returned over all documents
        it = LabeledLineSentence(train_data, names_train)

        model = gensim.models.Doc2Vec(vector_size=300, min_count=4, alpha=0.025, min_alpha=0.025)
        model.build_vocab(it)

        #training of model
        for epoch in range(num_epochs):
            print('Iteration', str(epoch+1))
            model.train(it,  
                        total_examples=model.corpus_count,
                        epochs=num_epochs)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
            
        #saving the created model
        model.save('doc2vec.model')
        print("model saved")
    
    d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
    
    print("Generating embeddings...")
    train_features = [get_vector(d2v_model, text) for text in train_data]
    test_features = [get_vector(d2v_model, text) for text in test_data]
    
    if description:
        pd.concat([train_features, df_train['label']], axis=1).to_csv("data/doc2vec_train_description.csv", index=False)
        pd.concat([test_features, df_test['label']], axis=1).to_csv("data/doc2vec_test_description.csv", index=False)
    else:
        pd.concat([train_features, df_train['label']], axis=1).to_csv("data/doc2vec_train.csv", index=False)
        pd.concat([test_features, df_test['label']], axis=1).to_csv("data/doc2vec_test.csv", index=False)


# In[8]:


def get_sentovec(train=False, num_epochs=10, oversample=False, description=False):
    print("Reading the data...")
        
    if oversample:
        df_train = get_oversampled_train()
    else:
        df_train = pd.read_csv("data/train_raw.csv")
        
    df_test = pd.read_csv("data/test_raw.csv")
    
    text_train = df_train['transcript']
    names_train = df_train['name']
    text_test = df_test['transcript']
    names_test = df_test['name']
    
    train_sentences = [list(map(nltk.word_tokenize, list(map(remove_parentheses,
                                                         sent_tokenize(parag)))))
                   for parag in text_train]
    test_sentences = [list(map(nltk.word_tokenize, list(map(remove_parentheses,
                                                             sent_tokenize(parag))))) 
                      for parag in text_test]
    
    # Flatten the sentences
    train_sen_data = [sentence for text in train_sentences for sentence in [item for item in text]]
    test_sen_data = [sentence for text in test_sentences for sentence in [item for item in text]]

    if train:
        print("Sen2Vec Training...")
        # Expand labels
        names_train = []
        for i, text in enumerate(train_sentences):
            for j, _ in enumerate(range(len(text))):
                names_train.append(df_train['name'][i] + str(j))

        names_test = []
        for i, text in enumerate(test_sentences):
            for j, _ in enumerate(range(len(text))):
                names_test.append(df_test['name'][i] + str(j))

        #iterator returned over all documents
        it = LabeledLineSentence(train_sen_data, names_train)

        model = gensim.models.Doc2Vec(vector_size=300, min_count=4, alpha=0.025, min_alpha=0.025)
        model.build_vocab(it)

        #training of model
        for epoch in range(num_epochs):
            print('Iteration', str(epoch+1))
            model.train(it,  
                        total_examples=model.corpus_count,
                        epochs=num_epochs)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        #saving the created model
        model.save('sen2vec.model')
        print("model saved")
    
    s2v_model = gensim.models.doc2vec.Doc2Vec.load('sen2vec.model')
    
    print("Generating embeddings...")
    train_features = [get_vector(s2v_model, text) for text in train_sen_data]
    test_features = [get_vector(s2v_model, text) for text in test_sen_data]
    
    train_ready_features = []
    prev = 0
    for text in train_sentences:
        train_ready_features.append(train_features[prev:len(text)])
        prev = len(text)
        
    test_ready_features = []
    prev = 0
    for text in test_sentences:
        test_ready_features.append(test_features[prev:len(text)])
        prev = len(text)
        
    return train_ready_features, test_ready_features


# In[13]:


# split_data(test_frac=0.15, undersample=False)


# In[14]:


# create_tfidf(oversample=False, description=True)


# In[ ]:


# create_doctovec(train=True, num_epochs=30, oversample=True)


# In[ ]:


# train, test = get_sentovec(train=True, num_epochs=30, oversample=True)


# In[ ]:


# get_undersampled_train(threshold=200, window=10)


# In[ ]:


# df_train = pd.read_csv("data/train_raw.csv")
# df_test = pd.read_csv("data/test_raw.csv")


# In[ ]:


# # Poorly represented labels
# threshold = 50
# window = 10
# indices = df_train['label'].value_counts()[df_train['label'].value_counts() < threshold].index.tolist()


# In[ ]:


# new_df_train = df_train
# for ind in indices:
#     poor_df = df_train[df_train['label'] == ind].reset_index(drop=True)
#     to_add = np.random.randint(threshold-window, threshold+window) - len(poor_df)
#     trans_ids = np.random.choice(range(len(poor_df)), 
#                                  to_add, 
#                                  replace=True)
#     dfs_to_add = poor_df.iloc[trans_ids]
#     new_df_train = pd.concat([new_df_train, dfs_to_add]).reset_index(drop=True)
# # Shuffle result
# new_df_train = new_df_train.sample(frac=1).reset_index(drop=True)


# In[ ]:


# new_df_train['label'].hist()


# In[ ]:


# new_df_train['label'].value_counts()

