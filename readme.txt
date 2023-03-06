Please follow the steps below to run scripts:
1. pip3 install -r requirements
2. python3 bert_with_stopword_lemmatizer.py
Make sure the file Bert_with_stopword_lemmatizer.h5 is saved.
3a. python3 other_models.py
(for training other compared models)
3b. python3 server.py
(for starting localhost web ui, the first loading may take time)

Programme Structure
├── Bert_with_stopword_lemmatizer.h5
├── __pycache__
│   ├── bert_classifier.cpython-310.pyc
│   ├── bert_classifier.cpython-311.pyc
│   ├── bert_with_stopword_lemmatizer.cpython-310.pyc
│   ├── lemmetizer_stop_word.cpython-310.pyc
│   ├── lemmetizer_stop_word.cpython-311.pyc
│   ├── simplied_bert.cpython-310.pyc
│   ├── text_processing.cpython-310.pyc
│   └── text_processing.cpython-311.pyc
├── bert_classifier.py
├── bert_with_stopword_lemmatizer.py
├── input
│   ├── sample_labels.csv
│   ├── test_data.txt
│   ├── train_data.txt
│   └── val_data.txt
├── other_models.py
├── output.csv
├── readme.txt
├── requirements.txt
├── server.py
├── templates
│   └── index.html
├── test_prediction.csv
└── utils
    ├── lemmetizer_stop_word.py
    └── text_processing.py