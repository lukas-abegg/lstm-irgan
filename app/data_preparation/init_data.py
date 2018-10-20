import os
import sys
import app.parameters as params


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


def get_data():
    documents_data = __get_documents()
    queries_data, query_ids = __get_queries()
    ratings_data = __get_ratings()

    print('Found %s training data.' % len(ratings_data))

    return query_ids, ratings_data, documents_data, queries_data
