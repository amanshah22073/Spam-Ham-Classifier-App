import nltk
import re
from nltk.corpus import stopwords
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend('\n')
import string


def preprocessing_pipeline(text):
    text_1 = []
    for char in text:
        if char not in string.punctuation:
            text_1.append(char)
        cleaned_text = "".join(text_1)
    text_list = []
    text = str(cleaned_text)
    text_without_urls = re.sub(r'http\S+|www.\S+', '', cleaned_text)
    text_list.append(text_without_urls)
    results = []
    tokens = gensim.utils.simple_preprocess(cleaned_text)
    for token in tokens:
        if len(token) >= 3 and token not in stop_words:
            results.append(token)
    cleaned_text = " ".join(results)
    return cleaned_text    
        
