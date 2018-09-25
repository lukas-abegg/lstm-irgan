import time
import numpy as np

import app.parameters as params
import app.eval.current_bests_MQ2008 as bests
import app.eval.ndcg_MQ2008 as ndcg_evaluator
import app.eval.precision_MQ2008 as p_evaluator


def train(generator, discriminator, query_url_features, query_urls, query_index_urls, query_pos_train, query_pos_test):
        for g_epoch in range(params.GEN_TRAIN_EPOCHS):
            start_time = time.time()
            print('now_ G_epoch : ', str(g_epoch))

            for query in query_pos_train.keys():
                pos_list = query_pos_train[query]
                pos_set = set(pos_list)
                # all url
                all_list = query_index_urls[query]
                # all feature
                all_list_feature = [query_url_features[query][url] for url in all_list]
                all_list_feature = np.asarray(all_list_feature)
                # G generate all url prob
                prob = generator.get_prob(all_list_feature[np.newaxis, :])
                prob = prob[0]
                prob = prob.reshape([-1])
                # important sampling, change doc prob
                prob_IS = prob * (1.0 - params.GEN_LAMBDA)

                for i in range(len(all_list)):
                    if all_list[i] in pos_set:
                        prob_IS[i] += (params.GEN_LAMBDA / (1.0 * len(pos_list)))

                # G generate some url (5 * postive doc num)
                choose_index = np.random.choice(np.arange(len(all_list)), [5 * len(pos_list)], p=prob_IS)
                # choose url
                choose_list = np.array(all_list)[choose_index]
                # choose feature
                choose_feature = [query_url_features[query][url] for url in choose_list]
                # prob / importan sampling prob (loss => prob * reward * prob / importan sampling prob)
                choose_IS = np.array(prob)[choose_index] / np.array(prob_IS)[choose_index]
                choose_index = np.asarray(choose_index)
                choose_feature = np.asarray(choose_feature)
                choose_IS = np.asarray(choose_IS)
                # get reward((prob  - 0.5) * 2 )
                choose_reward = discriminator.get_preresult(choose_feature)
                # train
                generator.train(choose_feature[np.newaxis, :], choose_reward.reshape([-1])[np.newaxis, :],
                                choose_IS[np.newaxis, :])

            print("train end--- %s seconds ---" % (time.time() - start_time))

            p_5_val = p_evaluator.precision_at_k_measure(generator, query_pos_test, query_pos_train, query_url_features, 5)
            ndcg_val = ndcg_evaluator.ndcg_at_k_measure(generator, query_pos_test, query_pos_train, query_url_features, 5)

            if p_5_val > bests.p_val:
                bests.p_val = p_5_val
                bests.ndcg_val = ndcg_val
                generator.save_model(params.GAN_MODEL_BEST_FILE)

                print("Best:", "gen p@5 ", p_evaluator, "gen ndcg@5 ", ndcg_evaluator)

            elif p_5_val == bests.p_val:
                if ndcg_val > bests.ndcg_val:
                    bests.ndcg_val = ndcg_val
                    generator.save_model(params.GAN_MODEL_BEST_FILE)

        print("Best:", "generator p@5 ", bests.p_val, "generator ndcg@5 ", bests.ndcg_val)
