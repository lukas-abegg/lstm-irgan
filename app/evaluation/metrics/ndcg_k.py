import numpy as np

import app.evaluation.eval_utils as utils


def measure_ndcg_at_k(model, x_val, ratings_data, queries_data, documents_data, k, sess):
    ndcg = 0.0
    cnt = 0

    for query_id in x_val:
        # get all query specific ratings
        x_data, y_data, eval_queries, eval_documents, enough_data_for_eval = utils.get_query_specific_eval_data(query_id, ratings_data, queries_data, documents_data, k)

        if not enough_data_for_eval:
            continue

        # predict y-values for given x-values
        pred_scores = sess.run(model.pred_score, feed_dict={model.pred_data: [eval_queries, eval_documents]})

        pred_document_scores_order, rated_document_scores_order = utils.sort_pred_val_data(x_data, y_data, pred_scores)

        relevant_k_rated_docs = set(rated_document_scores_order[:5])
        relevant_doc_set = [doc_id for doc_id, rating_score in relevant_k_rated_docs]

        dcg = 0.0
        for i in range(0, k):
            doc_id, score = pred_document_scores_order[i]
            if doc_id in relevant_doc_set:
                dcg += (1 / np.log2(i + 2))
        idcg = np.sum(np.ones(k) / np.log2(np.arange(2, k + 2)))

        ndcg += (dcg / idcg)
        cnt += 1

    return ndcg / float(cnt)
