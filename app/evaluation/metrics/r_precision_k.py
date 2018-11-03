import app.evaluation.eval_utils as utils


def measure_r_precision_at_k(model, x_val, ratings_data, queries_data, documents_data, sess):
    p = 0.0
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

        num = 0.0
        len_r = len(pred_document_scores_order)
        for i in range(0, len_r):
            doc_id, score = pred_document_scores_order[i]
            if doc_id in relevant_doc_set:
                num += 1.0
        num /= (len_r * 1.0)

        p += num
        cnt += 1

    return p / float(cnt)
