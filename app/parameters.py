# Base dirs
WORKDIR = '/home/lukas/git-projects/lstm-irgan'
TEMP = WORKDIR + '/temp'

# Hyperparameter Optimization
USE_HYPERPARAM_OPT = False

# Embeddings / Tokenizer
WORD2VEC = WORKDIR + '/data/embeddings/GoogleNews-vectors-negative300.bin.gz'

MAX_SEQUENCE_LENGTH_QUERIES = 20000
MAX_SEQUENCE_LENGTH_DOCUMENTS = 20000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# NN General
WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.001
TEMPERATURE = 0.2
DROPOUT = 0.2

OPT_MIN_WEIGHT_DECAY = 0.01
OPT_MIN_LEARNING_RATE = 0.001
OPT_MIN_TEMPERATURE = 0.2
OPT_MIN_DROPOUT = 0.2

OPT_MAX_WEIGHT_DECAY = 0.01
OPT_MAX_LEARNING_RATE = 0.001
OPT_MAX_TEMPERATURE = 0.2
OPT_MAX_DROPOUT = 0.2

# Discriminator
DISC_TRAIN_EPOCHS = 30
DISC_TRAIN_GEN_EPOCHS = 100
DISC_HIDDEN_SIZE_LSTM = 64
DISC_HIDDEN_SIZE_DENSE = 46

DISC_BATCH_SIZE = 8

# Generator
GEN_TRAIN_EPOCHS = 10
GEN_HIDDEN_SIZE_LSTM = 64
GEN_HIDDEN_SIZE_DENSE = 46
GEN_BATCH_SIZE = 8
GEN_LAMBDA = 0.5

# Data
DATA_SOURCE = 'example'
DATA_SOURCE_EXAMPLE = 'example'

DOCUMENTS_DIR = WORKDIR + '/data/example/documents/'
QUERIES = WORKDIR + '/data/example/queries.txt'
LABELLED_DATA = WORKDIR + '/data/example/labelled_data.txt'

# Evaluation
EVAL_K = 5

# Save models
SAVED_MODEL_DISC_FILE = TEMP + "/disc_model.h5"
SAVED_MODEL_GEN_FILE = TEMP + "/gen_model.h5"

# Plotting
PLOTTED_MODEL_FILE = TEMP + "/plot_model.png"

# General
KFOLD_SPLITS = 5
USED_MODE = 'train'

TRAIN_MODE = 'train'
EVAL_MODE = 'eval'
PLOT_MODEL_MODE = 'print_model'



