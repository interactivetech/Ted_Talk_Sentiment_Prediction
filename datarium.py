import pandas as pd
import textacy


def split_data(test_frac=0.2):
    df = pd.read_csv("data/Merged_dataset.csv")
    df_test = df.sample(frac=test_frac)
    df_train = df.drop(df_test.index)
    
    df_train.to_csv("data/train_raw.csv", index=False)
    df_test.to_csv("data/test_raw.csv", index=False)


def create_bow():
    df_train = pd.read_csv("data/train_raw.csv")
    df_test = pd.read_csv("data/test_raw.csv")
    
    corpus_train = textacy.Corpus(lang='en', texts=df_train['transcript'].tolist())
    corpus_test = textacy.Corpus(lang='en', texts=df_test['transcript'].tolist())
    
    tokenized_docs_train = (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) for doc in corpus_train)
    tokenized_docs_test = (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) for doc in corpus_test)
    
    vectorizer = textacy.Vectorizer(apply_idf=False, min_df=4, max_df=0.95)
    bow_train = vectorizer.fit_transform(tokenized_docs_train)
    bow_test = vectorizer.transform(tokenized_docs_test)
    
    bow_train = pd.DataFrame(bow_train.toarray())
    bow_test = pd.DataFrame(bow_test.toarray())
    
    pd.concat([bow_train, df_train['label']], axis=1).to_csv("data/bow_train.csv", index=False)
    pd.concat([bow_test, df_test['label']], axis=1).to_csv("data/bow_test.csv", index=False)


def create_tfidf():
    df_train = pd.read_csv("data/train_raw.csv")
    df_test = pd.read_csv("data/test_raw.csv")
    
    corpus_train = textacy.Corpus(lang='en', texts=df_train['transcript'].tolist())
    corpus_test = textacy.Corpus(lang='en', texts=df_test['transcript'].tolist())
    
    tokenized_docs_train = (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) for doc in corpus_train)
    tokenized_docs_test = (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) for doc in corpus_test)
    
    vectorizer = textacy.Vectorizer(apply_idf=True, norm="l2", min_df=4, max_df=.95)
    tfidf_train = vectorizer.fit_transform(tokenized_docs_train)
    tfidf_test = vectorizer.transform(tokenized_docs_test)
    
    tfidf_train = pd.DataFrame(tfidf_train.toarray())
    tfidf_test = pd.DataFrame(tfidf_test.toarray())
    
    pd.concat([tfidf_train, df_train['label']], axis=1).to_csv("data/tfidf_train.csv", index=False)
    pd.concat([tfidf_test, df_test['label']], axis=1).to_csv("data/tfidf_test.csv", index=False)

