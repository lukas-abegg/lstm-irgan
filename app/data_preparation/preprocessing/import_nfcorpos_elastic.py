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
    for key, value in documents_data.items():
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
                "_id": query,
                "query": query,
                "text": text,
                "rating": rating
            })

        i = i + len(value.items())

    print("import ", i, "ratings")
    return actions


def __create_index(es, index_to_create):
    if es.indices.exists(index_to_create):
        print("deleting '%s' index..." % index_to_create)
        res = es.indices.delete(index=index_to_create)
        print(" response: '%s'" % res)

    print("creating '%s' index..." % index_to_create)

    request_body = {
        "settings"
        : {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }

    res = es.indices.create(index=index_to_create, body=request_body)
    print(" response: '%s'" % res)


def __fill_index(es, actions):
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

index_doc = 'documents'
index_query = 'queries'
index_rating = 'ratings'

__create_index(es_instance, index_doc)
__create_index(es_instance, index_query)
__create_index(es_instance, index_rating)

__fill_index(es_instance, __import_documents(index_doc, 'document'))
__fill_index(es_instance, __import_queries(index_query, 'query'))
__fill_index(es_instance, __import_ratings(index_rating, 'rating'))
