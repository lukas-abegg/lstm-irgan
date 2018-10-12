import tensorflow as tf

import best_practice.eval.metrices.ndcg_MQ2008 as ndcg_evaluator
import best_practice.eval.metrices.precision_MQ2008 as precision_evaluator
import best_practice.eval.metrices.map_MQ2008 as map_evaluator
import best_practice.eval.metrices.mrr_MQ2008 as mrr_evaluator


def evaluate(generator, query_url_features, query_pos_train, query_pos_test):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    p_1_best = precision_evaluator.precision_at_k_measure(sess, generator, query_pos_test, query_pos_train, query_url_features, 1)
    p_3_best = precision_evaluator.precision_at_k_measure(sess, generator, query_pos_test, query_pos_train, query_url_features, 3)
    p_5_best = precision_evaluator.precision_at_k_measure(sess, generator, query_pos_test, query_pos_train, query_url_features, 5)
    p_10_best = precision_evaluator.precision_at_k_measure(sess, generator, query_pos_test, query_pos_train, query_url_features, 10)

    ndcg_1_best = ndcg_evaluator.ndcg_at_k_measure(sess, generator, query_pos_test, query_pos_train, query_url_features, 1)
    ndcg_3_best = ndcg_evaluator.ndcg_at_k_measure(sess, generator, query_pos_test, query_pos_train, query_url_features, 3)
    ndcg_5_best = ndcg_evaluator.ndcg_at_k_measure(sess, generator, query_pos_test, query_pos_train, query_url_features, 5)
    ndcg_10_best = ndcg_evaluator.ndcg_at_k_measure(sess, generator, query_pos_test, query_pos_train, query_url_features, 10)

    map_best = map_evaluator.map_measure(sess, generator, query_pos_test, query_pos_train, query_url_features)

    mrr_best = mrr_evaluator.mrr_measure(sess, generator, query_pos_test, query_pos_train, query_url_features)

    print("Best ", "p@1 ", p_1_best, "p@3 ", p_3_best, "p@5 ", p_5_best, "p@10 ", p_10_best)
    print("Best ", "ndcg@1 ", ndcg_1_best, "ndcg@3 ", ndcg_3_best, "ndcg@5 ", ndcg_5_best, "p@10 ", ndcg_10_best)
    print("Best MAP ", map_best)
    print("Best MRR ", mrr_best)
