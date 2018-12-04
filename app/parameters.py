# Base dirs
WORKDIR = '/home/abeggluk/lstm-irgan_binary'
TEMP = WORKDIR + '/temp'

# Hyperparameter Optimization
USE_HYPERPARAM_OPT = False

# Embeddings / Tokenizer
USE_FASTTEXT_MODEL = False
FASTTEXT_BINARY = True
FASTTEXT = '/home/abeggluk/BioWordVec_PubMed_MIMICIII_d200.vec.bin'

MAX_SEQUENCE_LENGTH = 2000
MAX_NUM_WORDS = 2000
EMBEDDING_DIM = 200

# NN General
WEIGHT_DECAY = 0.025
LEARNING_RATE = 0.001
TEMPERATURE = 0.2
DROPOUT = 0.2

OPT_MIN_LEARNING_RATE = 0.001
OPT_MIN_TEMPERATURE = 0.2
OPT_MIN_DROPOUT = 0.2

OPT_MAX_LEARNING_RATE = 0.001
OPT_MAX_TEMPERATURE = 0.2
OPT_MAX_DROPOUT = 0.2

# Discriminator
DISC_TRAIN_EPOCHS = 1
DISC_TRAIN_GEN_EPOCHS = 1
DISC_HIDDEN_SIZE_LSTM = 64
DISC_HIDDEN_SIZE_DENSE = 46

DISC_BATCH_SIZE = 8

# Generator
GEN_TRAIN_EPOCHS = 1
GEN_HIDDEN_SIZE_LSTM = 64
GEN_HIDDEN_SIZE_DENSE = 46
GEN_BATCH_SIZE = 8
GEN_LAMBDA = 0.5

# Data
DATA_SOURCE = 'nfcorpus'
DATA_SOURCE_NFCORPUS = 'nfcorpus'
DATA_SOURCE_WIKICLIR = 'wikiclir'
DATA_SOURCE_EXAMPLE = 'example'

DOCUMENTS_DIR = WORKDIR + '/data/nfcorpus/all_docs/train.docs'
QUERIES = WORKDIR + '/data/nfcorpus/all_queries/train.all.queries'
LABELLED_DATA = WORKDIR + '/data/nfcorpus/all_qrels/3-2-1/train.3-2-1.qrel'

# Training
POS_TRAINING_DATA_PER_QUERY = 5
MAX_RELEVANCE = 3

# Evaluation
EVAL_K = 5
EVAL_N_DOCS = 1000

# Save models
SAVED_MODEL_DISC_FILE = TEMP + "/disc_model.h5"
SAVED_MODEL_GEN_FILE = TEMP + "/gen_model.h5"

# Plotting
PLOTTED_MODEL_FILE = TEMP + "/plot_model.png"

# General
USED_MODE = 'train'

TRAIN_MODE = 'train'
EVAL_MODE = 'eval'
PLOT_MODEL_MODE = 'print_model'



