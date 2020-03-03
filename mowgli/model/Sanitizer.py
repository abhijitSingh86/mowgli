import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize

def normalize(msg):
    # Remove special characters
    removedSpecialCharStr = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", msg)
    # Remove numbers
    removedNumbersStr = re.sub(r"\d+", "", removedSpecialCharStr)
    # Remove white spaces
    removedWhiteSpaceStr = re.sub(r"\s+", " ", removedNumbersStr)
    return removedWhiteSpaceStr.strip(' ').lower()

def tokenizer(msg):
    tokens = word_tokenize(msg)
    return tokens


def word_stemmer(tokens):
    stem_text = [PorterStemmer().stem(i) for i in tokens]
    return stem_text

def word_lemmatizer(tokens):
    lem_text = [WordNetLemmatizer().lemmatize(i) for i in tokens]
    return lem_text

def testNormalize():
    msg = " # 1000 milling ' data# ? got IT "
    cleaned = "milling data got it"
    ret = normalize(msg)

    print(cleaned)
    print(ret)
    print(cleaned == ret)

def testStemming():
    msg= " # 1000 mailing '  mailer data# ? got IT "
    cleaned= "mail, dat, got, it"
    testToken = ['rains', 'rainer', 'rain', 'rained']
    normalizedData = normalize(msg)
    tokens = tokenizer(normalizedData)
    word_stemmer(tokens)

    print(word_stemmer(testToken))
    print(cleaned == tokens)

def testLammetization():
    msg = " # 1000 mailing '  mailer data# ? got IT earthquake "
    cleaned = "mail, dat, got, it, earthquak"
    lemmatized= 'mail, dat, got, it, earthquake'
    normalizedData = normalize(msg)

    print( word_lemmatizer(word_stemmer(tokenizer(normalizedData))))

def sanitizeTokens(msg):
    return word_lemmatizer(word_stemmer(tokenizer(normalize(msg))))


if __name__ == "__main__":
    testLammetization()