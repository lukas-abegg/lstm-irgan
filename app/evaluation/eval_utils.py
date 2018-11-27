import numpy as np
import app.parameters as params


def get_query_specific_eval_data(query_id, ratings_data, queries_data, documents_data, k=0):
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

    eval_queries = np.asarray(eval_queries)
    eval_documents = np.asarray(eval_documents)

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
