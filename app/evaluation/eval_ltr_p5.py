from comet_ml import Experiment

import parameters as params
import pandas as pd


def evaluate(model, x_val, ratings_data, queries_data, documents_data, sess, experiment: Experiment):
    p_2 = 0
    p_3 = 0
    p_5 = 0
    p_10 = 0

    for j, gold_std_path in enumerate(params.TREC_CDS_2017_LABELLED_DATA_LTR):
        print("------------------------------------------------")
        print("Runs for set", j, ":", gold_std_path)
        print("------------------------------------------------")
        for i in range(5):
            print("Run evaluation round ", i)
            print("------------------------")

            ratings, pred_ratings = prepare_data(model, gold_std_path, queries_data, documents_data)

            p_2_best = measure_p_at_k_eval_all(ratings, pred_ratings, 2)
            experiment.log_metric("p@2", p_2_best)
            print("p@2", p_2_best)
            p_3_best = measure_p_at_k_eval_all(ratings, pred_ratings, 3)
            experiment.log_metric("p@3", p_3_best)
            print("p@3", p_3_best)
            p_5_best = measure_p_at_k_eval_all(ratings, pred_ratings, 5)
            experiment.log_metric("p@5", p_5_best)
            print("p@5", p_5_best)
            p_10_best = measure_p_at_k_eval_all(ratings, pred_ratings, 10)
            experiment.log_metric("p@10", p_10_best)
            print("p@10", p_10_best)

            if p_2_best >= p_2:
                p_2 = p_2_best

            if p_3_best >= p_3:
                p_3 = p_3_best

            if p_5_best >= p_5:
                p_5 = p_5_best

            if p_10_best >= p_10:
                p_10 = p_10_best

    print("------------------------")
    print("Final Best Result:")
    print("------------------------")
    experiment.log_metric("Best p@2", p_2)
    print("Best", "p@2", p_2)
    experiment.log_metric("Best p@3", p_3)
    print("Best", "p@3", p_3)
    experiment.log_metric("Best p@5", p_5)
    print("Best", "p@5", p_5)
    experiment.log_metric("Best p@10", p_10)
    print("Best", "p@10", p_10)


def sort_by_pred_merge_with_val_data(x_data, y_data, pred_scores):
    pred_document_scores = zip(x_data, pred_scores)
    pred_document_order = sorted(pred_document_scores, key=lambda x: x[1], reverse=True)

    rated_document_scores = zip(x_data, y_data)
    rated_document_order = sorted(rated_document_scores, key=lambda x: x[1], reverse=False)

    return pred_document_order, rated_document_order


def measure_p_at_k_eval_all(ratings, pred_ratings, k):
    p = 0
    cnt = 0

    for i in ratings.keys():
        if i in pred_ratings:
            rating_gold = ratings[i]
            rating_pred = pred_ratings[i]

            pred_document_order, rated_document_order = sort_by_pred_merge_with_val_data(rating_gold.keys(), rating_gold.values(), rating_pred.values())

            if len(pred_document_order) >= k:
                relevant_k_rated_docs = set(rated_document_order[:k])
                relevant_doc_set = [doc_id for doc_id, rating_score in relevant_k_rated_docs]

                num = 0.0
                for i in range(0, k):
                    doc_id, score = pred_document_order[i]
                    if doc_id in relevant_doc_set:
                        num += 1.0

                num /= (k * 1.0)

                p += num
                cnt += 1

    return p / float(cnt)


def prepare_data(model, gold_std_path, queries_data, documents_data):

    ratings = __get_ratings(gold_std_path)
    trials, queries = __get_documents_queries(gold_std_path)

    trials_with_content = get_content(trials, documents_data)
    queries_with_content = get_content(queries, queries_data)

    pred_queries, pred_query_ids, pred_trials, pred_trial_ids = prepare_prediction_data(ratings, trials_with_content, queries_with_content)

    pred_scores_all = model.get_prob(pred_queries, pred_trials)

    pred_ratings = split_probs_data_by_query(pred_query_ids, pred_trial_ids, pred_scores_all)

    return ratings, pred_ratings


def prepare_prediction_data(ratings, trials_with_content, queries_with_content):
    queries = []
    query_ids = []
    trials = []
    trial_ids = []

    for query_key in ratings.keys():
        for trial_key in ratings[query_key].keys():
            if query_key in queries_with_content:
                queries.append(queries_with_content[query_key])
                query_ids.append(query_key)
                trials.append(trials_with_content[trial_key])
                trial_ids.append(trial_key)

    return queries, query_ids, trials, trial_ids


def split_probs_data_by_query(pred_queries, pred_trials, pred_scores_all):
    ratings = {}

    for i in range(len(pred_queries)):
        topic_number = pred_queries[i]
        document = pred_trials[i]
        rating = float(pred_scores_all[i])

        if topic_number in ratings.keys():
            ratings[topic_number][document] = rating
        else:
            ratings[topic_number] = {document: rating}

    return ratings


def get_content(keys, data):
    content = {}

    for key in keys:
        if key in data:
            content[key] = data[key]

    return content


def __get_ratings(path):
    ratings = {}

    with open(path) as f:
        content = f.readlines()
        for line in content:
            values = line.split("\t")
            topic_number = values[0]
            document = values[2]
            rating = float(values[3])

            if topic_number in ratings.keys():
                ratings[topic_number][document] = rating
            else:
                ratings[topic_number] = {document: rating}
    return ratings


def __get_documents_queries(path):

    judgements = []

    with open(path) as f:
        judgements = judgements + f.readlines()

    judgements = [x.split("\t") for x in judgements]

    judgements = pd.DataFrame(judgements)
    judgements.columns = ['topic', 'q0', 'trial', 'rank', 'relevance', 'run']
    trials = judgements.trial.drop_duplicates().values
    topics = judgements.topic.drop_duplicates().values

    return trials, topics