import best_practice.parameters as params
from best_practice.gan.discriminator import Discriminator
from best_practice.gan.generator import Generator
import best_practice.data_preparation.mq2008.preprocessing_MQ2008 as mq
import best_practice.training.train_discriminator_MQ2008 as train_disc
import best_practice.training.train_gen_MQ2008 as train_gen
import best_practice.eval.eval_MQ2008 as evaluator
import cPickle


def __prepare_data(data):
    if data == params.MQ2008:
        return mq.get_data(), mq.get_file_size_train(), mq.get_file_size_test()


def __train():
    # get data
    (query_url_features, query_urls, query_index_urls, query_pos_train, query_pos_test), query_pos_train_file_size, query_pos_test_file_size = __prepare_data(params.MQ2008)

    # call discriminator, generator
    discriminator = Discriminator(params.FEATURE_SIZE, params.DISC_HIDDEN_SIZE, params.DISC_WEIGHT_DECAY, params.DISC_LEARNING_RATE)
    generator = Generator(params.FEATURE_SIZE, params.GEN_HIDDEN_SIZE, params.GEN_WEIGHT_DECAY, params.GEN_LEARNING_RATE, params.GEN_TEMPERATURE)

    print('start adversarial training')
    for epoch in range(params.TRAIN_EPOCHS):
        # Generator generate negative for Discriminator, then train Discriminator
        print('Training Discriminator ...')
        train_disc.train(generator, discriminator, query_url_features, query_urls,
                            query_index_urls, query_pos_train, query_pos_train_file_size)

        # Train Generator
        print('Training Generator ...')
        train_gen.train(generator, discriminator, query_url_features, query_urls,
                        query_index_urls, query_pos_train, query_pos_test)


def __evaluate():
    # get data
    (query_url_features, query_urls, query_index_urls, query_pos_train,
     query_pos_test), query_pos_train_file_size, query_pos_test_file_size = __prepare_data(params.MQ2008)

    # get generator
    generator = cPickle.load(open(params.GAN_MODEL_BEST_FILE))

    evaluator.evaluate(generator, query_url_features, query_pos_train, query_pos_test)


def main(mode):
    if params.TRAIN_MODE == mode:
        __train()
    elif params.EVAL_MODE == mode:
        __evaluate()
    else:
        print("unknown MODE")


if __name__ == '__main__':
    main(params.TRAIN_MODE)
