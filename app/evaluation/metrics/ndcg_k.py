import numpy as np

import evaluation.eval_utils as utils


def measure_ndcg_at_k(model, x_val, ratings_data, queries_data, documents_data, k, sess):
    ndcg = 0.0
    cnt = 0

    query_ids_all, x_data_all, y_data_all, eval_queries_all, eval_documents_all = utils.prepare_eval_data(x_val, ratings_data, queries_data, documents_data, k)

    pred_scores_all = model.get_prob(eval_queries_all, eval_documents_all)
    print("Prediction scores for ndcg_"+str(k)+": ", pred_scores_all)

    x_data_query, y_data_query, pred_scores_query = utils.split_probs_data_by_query(
        query_ids_all, x_data_all, y_data_all, pred_scores_all)

    for query_id in x_data_query.keys():

        x_data = x_data_query[query_id][:]
        y_data = y_data_query[query_id][:]
        pred_scores = pred_scores_query[query_id][:]

        pred_document_scores_order = utils.sort_by_pred_merge_with_val_data(x_data, y_data, pred_scores)
        ndcg += __ndcg_at_k(pred_document_scores_order, k)
        cnt += 1

    return ndcg / float(cnt)


"""Example from http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf"""


def __dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def __ndcg_at_k(r, k, method=1):
    dcg_max = __dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return __dcg_at_k(r, k, method) / dcg_max
