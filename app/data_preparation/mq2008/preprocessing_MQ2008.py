import app.data_preparation.mq2008.utils_MQ2008py as utils
import app.parameters as params


def get_data():
    print ('start - load MQ 2008')
    query_url_features, query_urls, query_index_urls = utils.load_all_query_url_feature(params.NORM_DATA_FILE, params.MQ2008_FEATURE_SIZE)
    query_pos_train = utils.get_query_pos(params.TRAIN_DATA_FILE)
    query_pos_test = utils.get_query_pos(params.TEST_DATA_FILE)
    print ('end - load MQ 2008')
    return query_url_features, query_urls, query_index_urls, query_pos_train, query_pos_test


def get_file_size_train():
    return utils.file_len(params.TRAIN_DATA_FILE)


def get_file_size_test():
    return utils.file_len(params.TEST_DATA_FILE)


def get_batch_data(file, index, size):
    return utils.get_batch_data(file, index, size)





