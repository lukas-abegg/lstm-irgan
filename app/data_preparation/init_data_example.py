import os
import sys
from nltk.corpus import stopwords

import parameters as params

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from data_preparation.preprocessing.preprocess_tokenizer import TokenizePreprocessor


def __get_documents():
    path = params.DOCUMENTS_DIR
    documents = {}
    if os.path.isdir(path):
        for idx, fname in enumerate(sorted(os.listdir(path))):
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='latin-1')
            t = f.read()
            i = t.find('\n')  # skip header
            if 0 < i:
                t = t[i:]
            documents[idx+1] = t
            f.close()
    return documents


def __get_queries():
    path = params.QUERIES
    queries = {}
    query_ids = []

    with open(path) as f:
        content = f.readlines()
        for line in content:
            values = line.split()
            id = int(values[0])
            text = values[1]
            queries[id] = text
            query_ids.append(id)
    return queries, query_ids


def __get_ratings():
    path = params.LABELLED_DATA
    ratings = {}

    with open(path) as f:
        content = f.readlines()
        for line in content:
            values = line.split()
            query = int(values[0])
            text = int(values[1])
            rating = float(values[2])

            if query in ratings.keys():
                ratings[query][text] = rating
            else:
                ratings[query] = {text: rating}

    return ratings


def __filter_stop_words(texts, stop_words):
    for i, text in enumerate(texts):
        new_text = [word for word in text.split() if word not in stop_words]
        texts[i] = ' '.join(new_text)
    return texts


def __init_tokenizer(text_data, max_sequence_length):
    texts = list(text_data.values())
    ids = list(text_data.keys())

    stop_words = set(stopwords.words("english"))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    texts = __filter_stop_words(texts, stop_words)

    # finally, vectorize the text samples into a 2D integer tensor
    preTokenizer = TokenizePreprocessor(rules=False)
    texts = preTokenizer.fit_transform(texts)
    tokenizer = Tokenizer(num_words=params.MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_sequence_length)

    text_data_sequenced = {}
    for i, text in enumerate(data):
        text_data_sequenced[ids[i]] = text

    return tokenizer, text_data_sequenced


def get_data():
    documents_data = __get_documents()
    queries_data, query_ids = __get_queries()
    ratings_data = __get_ratings()

    tokenizer_q, queries_data = __init_tokenizer(queries_data, params.MAX_SEQUENCE_LENGTH)
    tokenizer_d, documents_data = __init_tokenizer(documents_data, params.MAX_SEQUENCE_LENGTH)

    print('Found %s training data.' % len(ratings_data))

    return query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d
