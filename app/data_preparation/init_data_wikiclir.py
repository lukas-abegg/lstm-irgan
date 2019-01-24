import nltk
from nltk.corpus import stopwords

import parameters as params

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from data_preparation.preprocessing.preprocess_tokenizer import TokenizePreprocessor


def __get_documents():
    path = params.DOCUMENTS_DIR
    documents = {}
    doc_ids = []

    with open(path) as f:
        content = f.readlines()
        for line in content:
            values = line.split("\t", 1)
            id = int(values[0])
            text = values[1]
            documents[id] = text
            doc_ids.append(id)
    return documents, doc_ids


def __get_queries():
    path = params.QUERIES
    queries = {}
    query_ids = []

    with open(path) as f:
        content = f.readlines()
        for line in content:
            values = line.split("\t", 1)
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
            values = line.split("\t")
            query = int(values[0])
            text = int(values[2])
            rating = float(values[3])

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


def __init_tokenizer(text_data):
    texts = list(text_data.values())

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '’'])
    texts = __filter_stop_words(texts, stop_words)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return tokenizer


def __sequence_data(tokenizer, text_data, max_sequence_length):
    texts = list(text_data.values())
    ids = list(text_data.keys())

    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    text_data_sequenced = {}
    for i, text in enumerate(data):
        text_data_sequenced[ids[i]] = text

    return text_data_sequenced


def get_data():
    documents_data, doc_ids = __get_documents()
    queries_data, query_ids = __get_queries()
    ratings_data = __get_ratings()

    print('Fit Tokenizer')
    documents_queries_data = dict(documents_data.items() | queries_data.items())
    tokenizer = __init_tokenizer(documents_queries_data)
    tokenizer_q = tokenizer
    tokenizer_d = tokenizer

    print('Sequence queries')
    queries_data = __sequence_data(tokenizer, queries_data, params.MAX_SEQUENCE_LENGTH_QUERIES)
    print('Sequence documents')
    documents_data = __sequence_data(tokenizer, documents_data, params.MAX_SEQUENCE_LENGTH_DOCS)

    print('Found %s training data.' % len(ratings_data))

    return query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d
