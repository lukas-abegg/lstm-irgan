import nltk
from nltk.corpus import stopwords

import xml.etree.ElementTree as ET

from pathlib import Path
import os.path

import parameters as params

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from data_preparation.preprocessing.preprocess_tokenizer import TokenizePreprocessor


def __read_document(path):
    my_file = Path(path)
    if my_file.is_file():
        my_file.read_text()
    else:
        return ""


def __get_documents():
    path = params.TREC_CDS_2017_LABELLED_DATA
    documents = {}
    doc_ids = []

    with open(path) as f:
        content = f.readlines()
        for line in content:
            values = line.split(" ")
            id = values[2]
            # /000/00000/NCT00000102.txt
            folder = id.split("NCT")[1][:3]
            subfolder = id.split("NCT")[1][:5]
            path = params.TREC_CDS_2017_DOCUMENTS + "/" + folder + "/" + subfolder + "/" + id + ".txt"
            text = __read_document(path)
            documents[id] = text
            doc_ids.append(id)
    return documents, doc_ids


def __get_queries():
    path = params.TREC_CDS_2017_QUERIES
    topics = {}
    topic_ids = []

    tree = ET.parse(path)
    root = tree.getroot()

    for topic in root.iter('topic'):
        topic_number = (topic.attrib['number'])
        for child in topic:
            if child.tag == 'disease':
                disease = child.text
            if child.tag == 'gene':
                gene = child.text
            if child.tag == 'demographic':
                demographic = child.text
            if child.tag == 'other':
                other = child.text
        topics[topic_number] = " ".join([disease, gene, demographic, other])
        topic_ids.append(topic_number)
    return topics, topic_ids


def __get_ratings():
    path = params.TREC_CDS_2017_LABELLED_DATA
    ratings = {}

    with open(path) as f:
        content = f.readlines()
        for line in content:
            values = line.split(" ")
            topic_number = values[0]
            document = values[2]
            rating = float(values[3])

            if topic_number in ratings.keys():
                ratings[topic_number][document] = rating
            else:
                ratings[topic_number] = {document: rating}

    return ratings


def __filter_stop_words(texts, stop_words):
    for i, text in enumerate(texts):
        new_text = [word for word in text.split() if word not in stop_words]
        texts[i] = ' '.join(new_text)
    return texts


def __init_tokenizer(text_data, max_sequence_length):
    texts = list(text_data.values())
    ids = list(text_data.keys())

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '’'])
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


def __init_tokenizer_documents(text_data, max_sequence_length):
    texts = list(text_data.values())
    ids = list(text_data.keys())

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '’'])
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
    documents_data, doc_ids = __get_documents()
    queries_data, query_ids = __get_queries()
    ratings_data = __get_ratings()

    print('Tokenize queries')
    tokenizer_q, queries_data = __init_tokenizer(queries_data, params.MAX_SEQUENCE_LENGTH)
    print('Tokenize documents')
    tokenizer_d, documents_data = __init_tokenizer(documents_data, params.MAX_SEQUENCE_LENGTH)

    print('Found %s training data.' % len(ratings_data))

    return query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d
