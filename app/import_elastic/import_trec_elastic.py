from elasticsearch import helpers, Elasticsearch

import xml.etree.ElementTree as ET

from pathlib import Path

import pandas as pd


def __read_document(path):
    my_file = Path(path)
    if my_file.is_file():
        return my_file.read_text()
    else:
        return ""


def __get_documents():
    trials_judgements = LABELLED_DATA

    with open(trials_judgements) as f:
        judgements = f.readlines()

    judgements = [x.split() for x in judgements]

    judgements = pd.DataFrame(judgements)
    judgements.columns = ['topic', 'q0', 'trial', 'relevance']
    trials = judgements.trial.drop_duplicates().values

    path = DOCUMENTS_DIR
    documents = {}
    doc_ids = []

    for trial in trials:
        path_trial = path + "/" + trial + ".txt"

        print(path_trial)
        text = __read_document(path_trial)
        documents[trial] = text
        doc_ids.append(trial)

    return documents, doc_ids


def __get_queries():
    path = QUERIES
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
    path = LABELLED_DATA
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


def __import_documents(index_to_import_data, type_to_import):
    print("import ", len(documents_data.items()), "documents")
    for key, value in documents_data.items():
        yield {
            "_index": index_to_import_data,
            "_type": type_to_import,
            "_id": key,
            "text": value
        }


def __import_queries(index_to_import_data, type_to_import):
    print("import ", len(queries_data.items()), "queries")
    for key, value in queries_data.items():
        yield {
            "_index": index_to_import_data,
            "_type": type_to_import,
            "_id": key,
            "text": value
        }


def __import_ratings(index_to_import_data, type_to_import):
    i = 0
    actions = []
    for query, value in ratings_data.items():
        for text, rating in value.items():
            actions.append({
                "_index": index_to_import_data,
                "_type": type_to_import,
                "query": query,
                "document": text,
                "rating": rating
            })

        i = i + len(value.items())

    print("import ", i, "ratings")
    return actions


def __create_index(es, index_to_create, request_body):
    if es.indices.exists(index_to_create):
        print("deleting '%s' index..." % index_to_create)
        res = es.indices.delete(index=index_to_create)
        print(" response: '%s'" % res)

    print("creating '%s' index..." % index_to_create)

    res = es.indices.create(index=index_to_create, body=request_body)
    print(" response: '%s'" % res)


def __fill_index(es, actions):
    # bulk index the data
    print("bulk indexing...")
    helpers.bulk(client=es, actions=actions, chunk_size=100)


WORKDIR = '/mnt/fob-wbia-vol2/wbi_stud/abeggluk/lstm-irgan-disc_as_pred_shrinked'
TREC_CDS_2017_DATA = WORKDIR + '/data/trec_pm_2017/data'
DOCUMENTS_DIR = '/mnt/fob-wbia-vol2/wbi_stud/abeggluk/trec_2017/final'
QUERIES = TREC_CDS_2017_DATA + '/topics2017.xml'
LABELLED_DATA = TREC_CDS_2017_DATA + '/qrels-final-trials.txt'

ES_HOST = {
    "host": "localhost",
    "port": 9200
}

documents_data, doc_ids = __get_documents()
queries_data, query_ids = __get_queries()
ratings_data = __get_ratings()

# create ES client, create index
es_instance = Elasticsearch(hosts=[ES_HOST])

index_doc = 'documents_trec'
index_query = 'queries_trec'
index_rating = 'ratings_trec'

request_body_simple = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}

request_body_ratings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "rating": {
            "properties": {
                "query": {
                    "type": "keyword"
                },
                "document": {
                    "type": "keyword"
                },
                "rating": {
                    "type": "long"
                }
            }
        }
    }
}

__create_index(es_instance, index_doc, request_body_simple)
__create_index(es_instance, index_query, request_body_simple)
__create_index(es_instance, index_rating, request_body_ratings)

__fill_index(es_instance, __import_documents(index_doc, 'document'))
__fill_index(es_instance, __import_queries(index_query, 'query'))
__fill_index(es_instance, __import_ratings(index_rating, 'rating'))
