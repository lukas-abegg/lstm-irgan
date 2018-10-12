# Base dirs
WORKDIR = '/home/lukas/git-projects/lstm-irgan'
TEMP = WORKDIR + '/temp'
GAN_MODEL_BEST_FILE = TEMP + '/gan_model'

# Embeddings
WORD2VEC = WORKDIR + '/data/embeddings/GoogleNews-vectors-negative300.bin.gz'

EMBEDDINGS_MAX_NUM_WORDS = 20000
EMBEDDINGS_MAX_SEQ_LENGTH = 20000
EMBEDDINGS_DIM = 100

# Discriminator
DISC_TRAIN_EPOCHS = 30
DISC_TRAIN_GEN_EPOCHS = 100
DISC_HIDDEN_SIZE = 46
DISC_WEIGHT_DECAY = 0.01
DISC_LEARNING_RATE = 0.001
DISC_TEMPERATURE = 0.2
DISC_BATCH_SIZE = 8

# Generator
GEN_TRAIN_EPOCHS = 10
GEN_HIDDEN_SIZE = 46
GEN_WEIGHT_DECAY = 0.01
GEN_LEARNING_RATE = 0.001
GEN_TEMPERATURE = 0.2
GEN_BATCH_SIZE = 8
GEN_LAMBDA = 0.5

# Data
FEATURE_SIZE = 46
DOCUMENTS_DIR = WORKDIR + '/data/documents/'
QUERIES = WORKDIR + '/data/queries.txt'
LABELLED_DATA = WORKDIR + '/data/labelled_data.txt'

# General
KFOLD_SPLITS = 5
TRAIN_MODE = 'train'
EVAL_MODE = 'eval'



