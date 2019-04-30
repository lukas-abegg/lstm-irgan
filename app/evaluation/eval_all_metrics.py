import evaluation.metrics.precision_k as p_eval
import evaluation.metrics.r_precision_k as r_p_eval
import evaluation.metrics.ndcg_k as ndcg_eval
import evaluation.metrics.map_k as map_eval
import evaluation.metrics.mrr_k as mrr_eval

import evaluation.eval_utils as utils

from comet_ml import Experiment


def evaluate(model, x_val, ratings_data, queries_data, documents_data, sess, experiment: Experiment):
    k_max = 5

    p_1 = 0
    p_3 = 0
    p_5 = 0
    p_10 = 0

    ndcg_3 = 0
    ndcg_5 = 0
    ndcg_10 = 0

    r = 0
    map = 0
    mrr = 0

    for i in range(5):
        print("Run evaluation round ", i)
        print("------------------------")

        x_data_query, y_data_query, pred_scores_query = prepare_data(model, x_val, ratings_data, queries_data, documents_data, k_max)

        p_1_best = p_eval.measure_precision_at_k_eval_all(x_data_query, y_data_query, pred_scores_query, 1, sess)
        experiment.log_metric("p@1", p_1_best)
        print("p@1", p_1_best)
        p_3_best = p_eval.measure_precision_at_k_eval_all(x_data_query, y_data_query, pred_scores_query, 3, sess)
        experiment.log_metric("p@3", p_3_best)
        print("p@3", p_3_best)
        p_5_best = p_eval.measure_precision_at_k_eval_all(x_data_query, y_data_query, pred_scores_query, 5, sess)
        experiment.log_metric("p@5", p_5_best)
        print("p@5", p_5_best)
        p_10_best = p_eval.measure_precision_at_k_eval_all(x_data_query, y_data_query, pred_scores_query, 10, sess)
        experiment.log_metric("p@10", p_10_best)
        print("p@10", p_10_best)

        ndcg_3_best = ndcg_eval.measure_ndcg_at_k_eval_all(x_data_query, y_data_query, pred_scores_query, 3, sess)
        experiment.log_metric("ndcg@3", ndcg_3_best)
        print("ndcg@3", ndcg_3_best)
        ndcg_5_best = ndcg_eval.measure_ndcg_at_k_eval_all(x_data_query, y_data_query, pred_scores_query, 5, sess)
        experiment.log_metric("ndcg@5", ndcg_5_best)
        print("ndcg@5", ndcg_5_best)
        ndcg_10_best = ndcg_eval.measure_ndcg_at_k_eval_all(x_data_query, y_data_query, pred_scores_query, 10, sess)
        experiment.log_metric("ndcg@10", ndcg_10_best)
        print("ndcg@10", ndcg_10_best)

        r_best = r_p_eval.measure_r_precision_at_k_eval_all(x_data_query, y_data_query, pred_scores_query, 10, 10, sess)
        experiment.log_metric("R@" + str(len(ratings_data)), r_best)
        print("R@" + str(len(ratings_data)), r_best)

        map_best = map_eval.measure_map_eval_all(x_data_query, y_data_query, pred_scores_query, 10, sess)
        experiment.log_metric("MAP", map_best)
        print("MAP", map_best)

        mrr_best = mrr_eval.measure_mrr_eval_all(x_data_query, y_data_query, pred_scores_query, 10, sess)
        experiment.log_metric("MRR", mrr_best)
        print("MRR", mrr_best)

        if (p_5_best >= p_5) and (ndcg_5_best >= ndcg_5):
            p_1 = p_1_best
            p_3 = p_3_best
            p_5 = p_5_best
            p_10 = p_10_best

            ndcg_3 = ndcg_3
            ndcg_5 = ndcg_5
            ndcg_10 = ndcg_10

            r = r_best
            map = map_best
            mrr = mrr_best

    experiment.log_metric("Best p@1", p_1)
    print("Best", "p@1", p_1)
    experiment.log_metric("Best p@3", p_3)
    print("Best", "p@3", p_3)
    experiment.log_metric("Best p@5", p_5)
    print("Best", "p@5", p_5)
    experiment.log_metric("Best p@10", p_10)
    print("Best", "p@10", p_10)

    experiment.log_metric("Best ndcg@3", ndcg_3)
    print("Best", "ndcg@3", ndcg_3)
    experiment.log_metric("Best ndcg@5", ndcg_5)
    print("Best", "ndcg@5", ndcg_5)
    experiment.log_metric("Best ndcg@10", ndcg_10)
    print("Best", "ndcg@10", ndcg_10)

    experiment.log_metric("Best R@" + str(len(ratings_data)), r)
    print("Best R@" + str(len(ratings_data)), r)
    experiment.log_metric("Best MAP", map)
    print("Best MAP", map)
    experiment.log_metric("Best MRR", mrr)
    print("Best MRR", mrr)


def prepare_data(model, x_val, ratings_data, queries_data, documents_data, k):
    query_ids_all, x_data_all, y_data_all, eval_queries_all, eval_documents_all = utils.prepare_eval_data(x_val,
                                                                                                          ratings_data,
                                                                                                          queries_data,
                                                                                                          documents_data)

    pred_scores_all = model.get_prob(eval_queries_all, eval_documents_all)
    print("Prediction scores for p_" + str(k) + ": ", pred_scores_all, "| found scores =", len(pred_scores_all),
          " for ", len(eval_queries_all), " queries")

    x_data_query, y_data_query, pred_scores_query = utils.split_probs_data_by_query(
        query_ids_all, x_data_all, y_data_all, pred_scores_all)

    return x_data_query, y_data_query, pred_scores_query
