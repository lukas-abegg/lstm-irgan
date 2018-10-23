import numpy as np


def get_query_specific_eval_data(query_id, ratings_data, queries_data, documents_data, k=0):
    x_data = list(ratings_data[query_id].keys())
    y_data = list(ratings_data[query_id].values())

    if k != 0:
        enough_data_for_eval: bool = len(x_data) >= k
    else:
        enough_data_for_eval = True

    eval_queries = []
    eval_documents = []

    if enough_data_for_eval:
        eval_queries = [queries_data[query_id]] * len(x_data)
        eval_documents = [documents_data[doc_id] for doc_id in x_data]

    eval_queries = np.asarray(eval_queries)
    eval_documents = np.asarray(eval_documents)

    return x_data, y_data, eval_queries, eval_documents, enough_data_for_eval


def sort_pred_val_data(x_data, y_data, pred_scores):
    pred_document_scores = zip(x_data, pred_scores)
    pred_document_scores_order = sorted(pred_document_scores, key=lambda x: x[1], reverse=True)

    rated_document_scores = zip(x_data, y_data)
    rated_document_scores_order = sorted(rated_document_scores, key=lambda x: x[1], reverse=True)

    return pred_document_scores_order, rated_document_scores_order


