import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import contractions
import re
import transformers
from transformers import BertTokenizer, TFBertForSequenceClassification
import keras_metrics
from utils.lemmetizer_stop_word import Lemmatizer_stop_word
from sklearn.metrics import classification_report


train = pd.read_csv(r'input/train_data.txt', names=['sentences', 'emotion'], sep=';')
val = pd.read_csv(r'input/val_data.txt', names=['sentences', 'emotion'], sep=';')
test = pd.read_csv(r'input/test_data.txt', names=['sentences', 'emotion'], sep=';')

train['sentences'] = train['sentences'].apply(lambda x: Lemmatizer_stop_word(x))
val['sentences'] = val['sentences'].apply(lambda x: Lemmatizer_stop_word(x))
test['sentences'] = test['sentences'].apply(lambda x: Lemmatizer_stop_word(x))
train.head()

max_length=43


from transformers import AutoTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense

tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')
bert=TFBertModel.from_pretrained('bert-base-cased')
x_train = tokenizer(
    [x.split() for x in train['sentences']],
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    is_split_into_words=True,
    verbose = True)


x_val = tokenizer(
    [x.split() for x in val['sentences']],
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    is_split_into_words=True,
    verbose = True)

lb = LabelEncoder()
labels_train=lb.fit(train.loc[:,'emotion'].to_list())
labels_train=lb.transform(train.loc[:,'emotion'].to_list())
labels_val=lb.transform(val.loc[:,'emotion'].to_list())


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(79)

input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

embeddings = bert(input_ids,attention_mask = input_mask)[0] #(0 is the last hidden states,1 is the pooler_output)
x = tf.keras.layers.GlobalMaxPool1D()(embeddings)
x = Dense(138, activation='elu',kernel_initializer='GlorotNormal')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = Dense(28,activation = 'elu',kernel_initializer='GlorotNormal')(x)

output = Dense(6,activation = 'softmax')(x)
    
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=output)
model.layers[2].trainable = True


opt = Adam(
    learning_rate=5e-05, 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy',keras_metrics.precision(),keras_metrics.recall()]) 

#'sparse_categorical_crossentropy' for not one-hot encoded features
# summarize the model
print(model.summary())

# fit the model
early_stopping_cb=keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)

history = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y =labels_train,
    validation_data = (
    {'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']}, labels_val
    ),
  epochs=3,
    batch_size=12,callbacks=[early_stopping_cb]
)

model.save_weights('Bert_with_stopword_lemmatizer.h5')
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0,1)
loss, accuracy, precision, recall = model.evaluate({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']}, labels_val)
predicted = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})
y_pred = predicted.argmax(axis=-1)

print(classification_report(val['emotion'],  lb.inverse_transform(y_pred)))
print('Accuracy: %f' % (accuracy*100))

#to visualize activation functions
for i, layer in enumerate (model.layers):
    print (i, layer)
    try:
        print ("    ",layer.activation)
    except AttributeError:
        print('   no activation attribute')
#specific info about each layer
for i in range(len(model.layers)):
    print(f'{i}   {model.layers[i]}: \n{model.layers[i].get_config()} \n')
#info about optimizers
model.optimizer.get_config()

x_test = tokenizer(
    [x.split() for x in test['sentences']],
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    is_split_into_words=True,
    verbose = True)

int2label = {
  0: "anger",
  1: "fear",
    2:"joy",
    3:"love",
    4:"sadness",
    5:"surprise"
}

df_result = pd.read_csv(r'input/sample_labels.csv')

print(model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}))
toList=np.argmax(model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}), axis=-1)
for index, item in enumerate(toList):
    df_result.loc[index,'class']=lb.inverse_transform([item])[0]
df_result.to_csv('output.csv', index=False)
