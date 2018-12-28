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
            id = values[0]
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
            id = values[0]
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
            query = values[0]
            text = values[2]
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


def __init_tokenizer(text_data, max_sequence_length, max_num_words):
    texts = list(text_data.values())
    ids = list(text_data.keys())

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '’'])
    texts = __filter_stop_words(texts, stop_words)

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=max_num_words)
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
    documents_data, doc_ids = __get_documents()
    queries_data, query_ids = __get_queries()
    ratings_data = __get_ratings()

    print('Tokenize queries')
    tokenizer_q, queries_data = __init_tokenizer(queries_data, params.MAX_SEQUENCE_LENGTH_QUERIES, params.MAX_NUM_WORDS_QUERIES)
    print('Tokenize documents')
    tokenizer_d, documents_data = __init_tokenizer(documents_data, params.MAX_SEQUENCE_LENGTH_DOCS, params.MAX_NUM_WORDS_DOCS)

    print('Found %s training data.' % len(ratings_data))

    return query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d
