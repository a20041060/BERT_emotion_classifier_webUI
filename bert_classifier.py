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
from keras.utils.vis_utils import plot_model

from utils.lemmetizer_stop_word import Lemmatizer_stop_word
from utils.text_processing import text_processing

max_length=43
lb,labels_train,labels_train,labels_val = text_processing()

opt = Adam(
    learning_rate=5e-05, # works well with BERTs
    epsilon=1e-08,
    clipnorm=1.0)

#rebuild the model to load the weights
tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')
bert=TFBertModel.from_pretrained('bert-base-cased')

input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

embeddings = bert(input_ids,attention_mask = input_mask)[0] #(0 is the last hidden states,1 is the pooler_output)
x = tf.keras.layers.GlobalMaxPool1D()(embeddings)
x = Dense(138, activation='elu',kernel_initializer='GlorotNormal')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = Dense(28,activation = 'elu',kernel_initializer='GlorotNormal')(x)

output = Dense(6,activation = 'softmax')(x)


def init_model():
    model_saved = tf.keras.Model(inputs=[input_ids, input_mask], outputs=output)
    model_saved.layers[2].trainable = True
    model_saved.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

    model_saved.load_weights('Bert_with_stopword_lemmatizer.h5')

    model_saved.optimizer.get_config()
    return model_saved

def classifier(y,model_saved):
    y_s=pd.Series([y])
    y_lemm=y_s.apply(lambda x: Lemmatizer_stop_word(x))
    y_tok = tokenizer(
        [x.split() for x in y_lemm],
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',  #only for sentence prediction 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        is_split_into_words=True,
        verbose = True)
    y_prob=model_saved.predict({'input_ids':y_tok['input_ids'],'attention_mask':y_tok['attention_mask']})*100

    print(y_prob)
    #y_tok
    class_label=y_prob.argmax(axis=-1)
    lb.inverse_transform(class_label) #from class to label

    print({'input_ids':y_tok['input_ids'],'attention_mask':y_tok['attention_mask']}) #bert input parameters

    print(lb.inverse_transform(class_label))
    return lb.inverse_transform(class_label)[0]
