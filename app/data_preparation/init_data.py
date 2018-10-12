import os
import sys
import pandas as pd
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
    with open(path) as f:
        content = f.readlines()
        for line in content:
            values = line.split()
            id = int(values[0])
            text = values[1]
            queries[id] = text
    return queries


def __get_labels():
    path = params.LABELLED_DATA
    labels = []
    with open(path) as f:
        content = f.readlines()
        for line in content:
            labels.append((line.split()))
    return labels


def get_data():
    documents_dict = __get_documents()
    queries_dict = __get_queries()
    labels = __get_labels()

    labels_data = pd.DataFrame(labels)

    x_data = labels_data.drop(columns=[2])
    x_data = x_data.rename(columns={0: 'query_id', 1: 'doc_id'})

    y_data = labels_data.drop(columns=[0, 1])
    y_data = y_data.rename(columns={2: 'relevance'})

    documents_data = pd.DataFrame.from_dict(documents_dict, orient='index')
    documents_data = documents_data.rename(columns={0: 'text'})
    queries_data = pd.DataFrame.from_dict(queries_dict, orient='index')
    queries_data = queries_data.rename(columns={0: 'query'})

    print('Found %s training data.' % len(y_data))

    return x_data, y_data, documents_data, queries_data
