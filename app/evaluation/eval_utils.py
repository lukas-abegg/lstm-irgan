import numpy as np
import parameters as params


def __get_query_specific_eval_data(query_id, ratings_data, queries_data, documents_data, k=0):
    x_data = list(ratings_data[query_id].keys())
    y_data = list(ratings_data[query_id].values())

    if k != 0:
        enough_data_for_eval: bool = len(x_data) >= k
    else:
        enough_data_for_eval = True

    eval_queries = []
    eval_documents = []

    n_eval = params.EVAL_N_DOCS
    if len(x_data) < n_eval:
        n_candidates = n_eval - len(x_data)
    else:
        n_candidates = 0

    if enough_data_for_eval:
        candidates = __get_candidate_docs(x_data, documents_data, n_candidates)
        x_data.extend(candidates)
        y_data.extend([0.0] * n_candidates)

        eval_queries = [queries_data[query_id]] * n_eval
        eval_documents = [documents_data[doc_id] for doc_id in x_data]

    return x_data, y_data, eval_queries, eval_documents, enough_data_for_eval


def __get_candidate_docs(x_pos_list, documents_data, n_candidates):
    # get all other ratings
    docs_pos_ids = np.unique(x_pos_list)
    candidate_list = []
    for doc_id in documents_data.keys():
        if doc_id not in docs_pos_ids:
            candidate_list.append(doc_id)

    candidate_list = np.random.choice(candidate_list, n_candidates)

    return candidate_list


def split_probs_data_by_query(query_ids_all, x_data_all, y_data_all, pred_scores_all):
    x_data_query = {}
    y_data_query = {}
    pred_scores_query = {}

    for i, query_id in enumerate(query_ids_all):

        if 0 <= i < len(x_data_all):
            x_val = x_data_all[i]
        else:
            x_val = 0

        if query_id in x_data_query.keys():
            x_data_query[query_id] = np.append(x_data_query[query_id], x_val)
        else:
            x_data_query[query_id] = [x_val]

        if 0 <= i < len(y_data_all):
            y_val = y_data_all[i]
        else:
            y_val = 0

        if query_id in y_data_query.keys():
            y_data_query[query_id] = np.append(y_data_query[query_id], y_val)
        else:
            y_data_query[query_id] = [y_val]

        if 0 <= i < len(pred_scores_all):
            pred_score = pred_scores_all[i]
        else:
            pred_score = 0

        if query_id in pred_scores_query.keys():
            pred_scores_query[query_id] = np.append(pred_scores_query[query_id], pred_score)
        else:
            pred_scores_query[query_id] = [pred_score]

    return x_data_query, y_data_query, pred_scores_query


def sort_pred_val_data(x_data, y_data, pred_scores):
    pred_document_scores = zip(x_data, pred_scores)
    pred_document_scores_order = sorted(pred_document_scores, key=lambda x: x[1], reverse=True)

    rated_document_scores = zip(x_data, y_data)
    rated_document_scores_order = sorted(rated_document_scores, key=lambda x: x[1], reverse=True)

    return pred_document_scores_order, rated_document_scores_order


def sort_by_pred_merge_with_val_data(x_data, y_data, pred_scores):
    pred_document_scores = zip(x_data, pred_scores)
    pred_document_scores_order = sorted(pred_document_scores, key=lambda x: x[1], reverse=True)

    rated_document_scores = zip(x_data, y_data)
    rated_document_scores = dict(rated_document_scores)
    document_scores_order = [rated_document_scores[doc[0]] for doc in pred_document_scores_order]

    return document_scores_order


def prepare_eval_data(x_val, ratings_data, queries_data, documents_data, k):
    query_ids_all = []
    x_data_all = []
    y_data_all = []
    eval_queries_all = []
    eval_documents_all = []

    for query_id in x_val:
        # get all query specific ratings
        x_data, y_data, eval_queries, eval_documents, enough_data_for_eval = __get_query_specific_eval_data(query_id, ratings_data, queries_data, documents_data, k)

        if not enough_data_for_eval:
            continue

        x_data_all.extend(x_data)
        y_data_all.extend(y_data)
        eval_queries_all.extend(eval_queries)
        eval_documents_all.extend(eval_documents)
        query_ids_all.extend([query_id] * len(x_data))

    # predict y-values for given x-values
    eval_queries_all = np.asarray(eval_queries_all)
    eval_documents_all = np.asarray(eval_documents_all)

    return query_ids_all, x_data_all, y_data_all, eval_queries_all, eval_documents_all
