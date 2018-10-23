# Base dirs
WORKDIR = '/home/lukas/git-projects/lstm-irgan'
TEMP = WORKDIR + '/temp'

# Embeddings / Tokenizer
WORD2VEC = WORKDIR + '/data/embeddings/GoogleNews-vectors-negative300.bin.gz'

MAX_SEQUENCE_LENGTH_QUERIES = 20000
MAX_SEQUENCE_LENGTH_DOCUMENTS = 20000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# Discriminator
DISC_TRAIN_EPOCHS = 30
DISC_TRAIN_GEN_EPOCHS = 100
DISC_HIDDEN_SIZE_LSTM = 64
DISC_HIDDEN_SIZE_DENSE = 46
DISC_WEIGHT_DECAY = 0.01
DISC_LEARNING_RATE = 0.001
DISC_TEMPERATURE = 0.2
DISC_BATCH_SIZE = 8

# Generator
GEN_TRAIN_EPOCHS = 10
GEN_HIDDEN_SIZE_LSTM = 64
GEN_HIDDEN_SIZE_DENSE = 46
GEN_WEIGHT_DECAY = 0.01
GEN_LEARNING_RATE = 0.001
GEN_TEMPERATURE = 0.2
GEN_BATCH_SIZE = 8
GEN_LAMBDA = 0.5

# Data
DOCUMENTS_DIR = WORKDIR + '/data/documents/'
QUERIES = WORKDIR + '/data/queries.txt'
LABELLED_DATA = WORKDIR + '/data/labelled_data.txt'

# Evaluation
EVAL_K = 5

# General
KFOLD_SPLITS = 5
TRAIN_MODE = 'train'
EVAL_MODE = 'eval'



