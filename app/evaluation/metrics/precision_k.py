import evaluation.eval_utils as utils


def measure_precision_at_k(model, x_val, ratings_data, queries_data, documents_data, k, sess):
    p = 0.0
    cnt = 0

    query_ids_all, x_data_all, y_data_all, eval_queries_all, eval_documents_all = utils.prepare_eval_data(x_val, ratings_data, queries_data, documents_data, k)

    pred_scores_all = model.get_prob(eval_queries_all, eval_documents_all)
    print("Prediction scores for p_k: ", pred_scores_all, "| found scores =", len(pred_scores_all), " for ", len(eval_queries_all), " queries")

    x_data_query, y_data_query, pred_scores_query = utils.split_probs_data_by_query(
        query_ids_all, x_data_all, y_data_all, pred_scores_all)

    for query_id in x_data_query.keys():

        x_data = x_data_query[query_id][:]
        y_data = y_data_query[query_id][:]
        pred_scores = pred_scores_query[query_id][:]

        pred_document_scores_order, rated_document_scores_order = utils.sort_pred_val_data(x_data, y_data, pred_scores)

        relevant_k_rated_docs = set(rated_document_scores_order[:k])
        relevant_doc_set = [doc_id for doc_id, rating_score in relevant_k_rated_docs]

        num = 0.0
        for i in range(0, k):
            doc_id, score = pred_document_scores_order[i]
            if doc_id in relevant_doc_set:
                num += 1.0

        num /= (k * 1.0)

        p += num
        cnt += 1

    return p / float(cnt)
