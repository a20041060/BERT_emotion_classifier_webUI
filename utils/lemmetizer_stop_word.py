import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def Lemmatizer_stop_word(sentence):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer() 
    sentence = re.sub('[^A-z]', ' ', sentence)
    negative = ['not', 'neither', 'nor', 'but', 'however',
                'although', 'nonetheless', 'despite', 'except',
                        'even though', 'yet','unless']
    stop_words = [z for z in stop_words if z not in negative]
    preprocessed_tokens = [lemmatizer.lemmatize(contractions.fix(temp.lower())) for temp in sentence.split() if temp not in stop_words] #lemmatization
    return ' '.join([x for x in preprocessed_tokens]).strip()