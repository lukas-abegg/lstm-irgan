from elasticsearch import helpers, Elasticsearch


def __get_documents():
    path = DOCUMENTS_DIR
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
    path = QUERIES
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
    path = LABELLED_DATA
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


def __import_documents(index, type):
    print("import ", len(documents_data.items()), "documents")
    for key, value in documents_data.items():
        yield {
                "_index": index,
                "_type": type,
                "_id": key,
                "text": value
        }


def __import_queries(index, type):
    print("import ", len(queries_data.items()), "queries")
    for key, value in documents_data.items():
        yield {
                "_index": index,
                "_type": type,
                "_id": key,
                "text": value
        }


def __import_ratings(index, type):
    print("import ", len(documents_data.items()), "ratings")
    for key, value in documents_data.items():
        yield {
                "_index": index,
                "_type": type,
                "_id": key,
                "text": value
        }


def __create_index(es, index, request_body, actions):
    if es.indices.exists(index):
        print("deleting '%s' index..." % index)
        res = es.indices.delete(index=index)
        print(" response: '%s'" % res)

    print("creating '%s' index..." % index)
    res = es.indices.create(index=index, body=request_body)
    print(" response: '%s'" % res)

    # bulk index the data
    print("bulk indexing...")
    helpers.bulk(client=es, actions=actions, chunk_size=100)


WORKDIR = '/home/lukas/git-projects/lstm-irgan'
DOCUMENTS_DIR = WORKDIR + '/data/nfcorpus/all_docs/train.docs'
QUERIES = WORKDIR + '/data/nfcorpus/all_queries/train.all.queries'
LABELLED_DATA = WORKDIR + '/data/nfcorpus/all_qrels/3-2-1/train.3-2-1.qrel'

ES_HOST = {
    "host": "localhost",
    "port": 9200
}

documents_data, doc_ids = __get_documents()
queries_data, query_ids = __get_queries()
ratings_data = __get_ratings()

# create ES client, create index
es_instance = Elasticsearch(hosts=[ES_HOST])

request_body_basic = {
    "settings"
    : {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}

__create_index(es_instance, 'documents', request_body_basic, __import_documents('documents', 'document'))
__create_index(es_instance, 'queries', request_body_basic, __import_queries('queries', 'query'))
__create_index(es_instance, 'ratings', request_body_basic, __import_ratings('ratings', 'rating'))
