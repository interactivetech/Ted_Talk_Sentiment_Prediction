import pandas as pd
import textacy
import json
from utils import *


def merge_transcripts():
    def combine_list_of_dicts(arr):
        dict_ratings = {}
        for a in arr:
            if a["id"] not in dict_ratings:
                dict_ratings[a["id"]] = a

        return dict_ratings

    def getTEDRating(string):
        arr = json.loads(string.replace("'", '\"'))
        d = combine_list_of_dicts(arr)
        return sorted(d.items(), key=lambda x: x[1]["count"], reverse=True)[0][1]['id']

    print("Merging transcripts")
    df1 = pd.read_csv("data/transcripts.csv")
    df2 = pd.read_csv("data/ted_main.csv")
    merged = df1.merge(df2, on="url")
    labels = [getTEDRating(x) for x in merged['ratings'].tolist()]
    merged['label'] = labels
    merged.to_csv("data/merged_dataset.csv", index=False)


def split_data(test_frac=0.2):
    print("Splitting the data")
    df = pd.read_csv("data/merged_dataset.csv")
    df_test = df.sample(frac=test_frac, random_state=42)
    df_train = df.drop(df_test.index)
    
    df_train.to_csv("data/train_raw.csv", index=False)
    df_test.to_csv("data/test_raw.csv", index=False)


def create_bow_tfidf():
    print("Creating BOW")
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

    # tf-idf
    tokenized_docs_train = (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) for doc in corpus_train)
    tokenized_docs_test = (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True) for doc in corpus_test)

    vectorizer = textacy.Vectorizer(apply_idf=True, norm="l2", min_df=4, max_df=.95)
    tfidf_train = vectorizer.fit_transform(tokenized_docs_train)
    tfidf_test = vectorizer.transform(tokenized_docs_test)

    tfidf_train = pd.DataFrame(tfidf_train.toarray())
    tfidf_test = pd.DataFrame(tfidf_test.toarray())

    pd.concat([tfidf_train, df_train['label']], axis=1).to_csv("data/tfidf_train.csv", index=False)
    pd.concat([tfidf_test, df_test['label']], axis=1).to_csv("data/tfidf_test.csv", index=False)


if __name__ == "__main__":
    start = check_time()
    merge_transcripts()
    check_time(start, "merging")

    start = check_time()
    split_data(test_frac=0.2)
    check_time(start, "splitting")

    start = check_time()
    create_bow_tfidf()
    check_time(start, "bow_tf-idf")

