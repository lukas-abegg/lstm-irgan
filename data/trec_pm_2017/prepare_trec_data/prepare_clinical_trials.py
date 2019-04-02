import pandas as pd
import numpy as np

from nltk.corpus import stopwords

import xml.etree.ElementTree as ET

from pathlib import Path

import shutil

trials_judgements = "../data/qrels-final-trials2.txt"
trials_topics = "../data/topics2017.xml"
trials_documents = "../data/documents"

trials_documents_dest = "../data/documents/final"

with open(trials_judgements) as f:
    judgements = f.readlines()

judgements = [x.split() for x in judgements]

judgements = pd.DataFrame(judgements)
judgements.columns = ['topic', 'q0', 'trial', 'relevance']
trials = judgements.trial.values


def __read_document(path):
    my_file = Path(path)
    if my_file.is_file():
        return my_file.read_text()
    else:
        return ""


def __copy_large_file(src, dest, buffer_size=16000):
    with open(src, 'rb') as fsrc:
        with open(dest, 'wb') as fdest:
            shutil.copyfileobj(fsrc, fdest, buffer_size)


for id in trials:
    # /000/00000/NCT00000102.txt
    folder = id.split("NCT")[1][:3]
    print(folder)
    subfolder = id.split("NCT")[1][:5]
    print(subfolder)

    source = trials_documents + "/" + folder + "/" + subfolder + "/" + id + ".txt"
    dest = trials_documents_dest + "/" + id + ".txt"
    print(source)
    print(dest)
    __copy_large_file(source, dest)

