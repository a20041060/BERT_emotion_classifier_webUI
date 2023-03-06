from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import contractions
import re
import numpy as np

from transformers import AutoTokenizer, TFBertModel

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from utils.lemmetizer_stop_word import Lemmatizer_stop_word

def text_processing():
    train = pd.read_csv(r'input/train_data.txt', names=['sentences', 'emotion'], sep=';')
    val = pd.read_csv(r'input/val_data.txt', names=['sentences', 'emotion'], sep=';')
    test = pd.read_csv(r'input/test_data.txt', names=['sentences', 'emotion'], sep=';')
    train.head()

    train['sentences'] = train['sentences'].apply(lambda x: Lemmatizer_stop_word(x))
    val['sentences'] = val['sentences'].apply(lambda x: Lemmatizer_stop_word(x))
    test['sentences'] = test['sentences'].apply(lambda x: Lemmatizer_stop_word(x))

    lb = LabelEncoder()
    labels_train=lb.fit(train.loc[:,'emotion'].to_list())
    labels_train=lb.transform(train.loc[:,'emotion'].to_list())
    labels_val=lb.transform(val.loc[:,'emotion'].to_list())
    return lb, labels_train, labels_train, labels_val