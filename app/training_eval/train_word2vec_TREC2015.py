import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
import tensorflow as tf

def tokenize(corpus):
    """Tokenize the corpus text.
    :param corpus: list containing a string of text (example: ["I like playing football with my friends"])
    :return corpus_tokenized: indexed list of words in the corpus, in the same order as the original corpus (the example above would return [[1, 2, 3, 4]])
    :return V: size of vocabulary
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    corpus_tokenized = tokenizer.texts_to_sequences(corpus)
    v = len(tokenizer.word_index)
    return corpus_tokenized, v

