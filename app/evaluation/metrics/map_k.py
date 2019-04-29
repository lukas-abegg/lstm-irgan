import numpy as np

import evaluation.eval_utils as utils


def measure_map(model, x_val, ratings_data, queries_data, documents_data, k, sess):
    rs = []

    query_ids_all, x_data_all, y_data_all, eval_queries_all, eval_documents_all = utils.prepare_eval_data(
        x_val, ratings_data, queries_data, documents_data, k)

    pred_scores_all = model.get_prob(eval_queries_all, eval_documents_all)
    print("Prediction scores for map_"+str(k)+": ", pred_scores_all)

    x_data_query, y_data_query, pred_scores_query = utils.split_probs_data_by_query(
        query_ids_all, x_data_all, y_data_all, pred_scores_all)

    for query_id in x_data_query.keys():

        x_data = x_data_query[query_id][:]
        y_data = y_data_query[query_id][:]
        pred_scores = pred_scores_query[query_id][:]

        pred_document_scores_order, rated_document_scores_order = utils.sort_pred_val_data(x_data, y_data, pred_scores)

        relevant_k_rated_docs = set(rated_document_scores_order[:k])
        relevant_doc_set = [doc_id for doc_id, rating_score in relevant_k_rated_docs]

        r = [0.0] * k
        for i in range(0, k):
            doc_id, score = pred_document_scores_order[i]
            if doc_id in relevant_doc_set:
                r[i] = 1.0
        rs.append(r)

    return np.mean([__average_precision(r) for r in rs])


def __precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def __average_precision(r):
    r = np.asarray(r)
    out = [__precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)
