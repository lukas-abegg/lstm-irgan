import nltk
from nltk.corpus import stopwords

import xml.etree.ElementTree as ET

import pandas as pd

from pathlib import Path

import parameters as params

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def __read_document(path):
    my_file = Path(path)
    if my_file.is_file():
        return my_file.read_text()
    else:
        print("File:", str(path), "is empty")
        return ""


def __get_documents():
    trials_judgements = params.TREC_CDS_2017_LABELLED_DATA_LTR

    judgements = []
    for path in trials_judgements:
        with open(path) as f:
            judgements = judgements + f.readlines()

    judgements = [x.split("\t") for x in judgements]

    judgements = pd.DataFrame(judgements)
    judgements.columns = ['topic', 'q0', 'trial', 'rank', 'relevance', 'run']
    trials = judgements.trial.drop_duplicates().values

    path = params.TREC_CDS_2017_DOCUMENTS_LTR
    documents = {}
    doc_ids = []

    for trial in trials:
        level1 = trial[3:6]
        level2 = trial[3:8]
        path_trial = path + "/" + level1 + "/" + level2 + "/" + trial + ".txt"

        text = __read_document(path_trial)
        documents[trial] = text
        doc_ids.append(trial)

    return documents, doc_ids


def __get_queries():
    path = params.TREC_CDS_2017_QUERIES_LTR
    topics = {}
    topic_ids = []

    tree = ET.parse(path)
    root = tree.getroot()

    for topic in root.iter('topic'):
        topic_number = (topic.attrib['number'])
        disease, gene, demographic, other = "", "", "", ""
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
    paths = params.TREC_CDS_2017_LABELLED_DATA_LTR
    ratings = {}

    for path in paths:
        with open(path) as f:
            content = f.readlines()
            for line in content:
                values = line.split("\t")
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
