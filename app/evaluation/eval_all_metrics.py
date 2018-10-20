import app.evaluation.metrics.precision_k as p_eval
import app.evaluation.metrics.ndcg_k as ndcg_eval
import app.evaluation.metrics.map_k as map_eval
import app.evaluation.metrics.mrr_k as mrr_eval


def evaluate(model, x_val, ratings_data, queries_data, documents_data, sess):

    p_1_best = p_eval.measure_precision_at_k(model, x_val, ratings_data, queries_data, documents_data, 1, sess)
    p_3_best = p_eval.measure_precision_at_k(model, x_val, ratings_data, queries_data, documents_data, 3, sess)
    p_5_best = p_eval.measure_precision_at_k(model, x_val, ratings_data, queries_data, documents_data, 5, sess)
    p_10_best = p_eval.measure_precision_at_k(model, x_val, ratings_data, queries_data, documents_data, 10, sess)

    ndcg_1_best = ndcg_eval.measure_ndcg_at_k(model, x_val, ratings_data, queries_data, documents_data, 1, sess)
    ndcg_3_best = ndcg_eval.measure_ndcg_at_k(model, x_val, ratings_data, queries_data, documents_data, 3, sess)
    ndcg_5_best = ndcg_eval.measure_ndcg_at_k(model, x_val, ratings_data, queries_data, documents_data, 5, sess)
    ndcg_10_best = ndcg_eval.measure_ndcg_at_k(model, x_val, ratings_data, queries_data, documents_data, 10, sess)

    map_best = map_eval.measure_map(model, x_val, ratings_data, queries_data, documents_data, sess)

    mrr_best = mrr_eval.measure_mrr(model, x_val, ratings_data, queries_data, documents_data, sess)

    print("Best ", "p@1 ", p_1_best, "p@3 ", p_3_best, "p@5 ", p_5_best, "p@10 ", p_10_best)
    print("Best ", "ndcg@1 ", ndcg_1_best, "ndcg@3 ", ndcg_3_best, "ndcg@5 ", ndcg_5_best, "p@10 ", ndcg_10_best)
    print("Best MAP ", map_best)
    print("Best MRR ", mrr_best)
