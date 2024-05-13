from langdetect import detect
import pandas as pd
import csv
import re
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords' , quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import time
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


lematizer = WordNetLemmatizer()
spelling = SpellChecker()
#function to clean and correct comments
def preprocess(comment):
    comment = re.sub(r'[^A-Za-z\s]', '', comment)

    cspellwords = [spelling.correction(word) if spelling.correction(word) is not None else word for word in comment.split()]
    token = word_tokenize(' '.join(cspellwords))

    stop_words = set(stopwords.words('english'))
    normalized_words = [word.lower() for word in token if word.lower() not in stop_words]

    lemmatized_words = []
    for word, tag in nltk.pos_tag(normalized_words):
        pos = tag[0].lower()  # Convert POS tag to WordNet format
        pos = pos if pos in ['a', 'r', 'n', 'v'] else 'n' 

        lemma = lematizer.lemmatize(word, pos=pos)
        lemmatized_words.append(lemma)

    print(lemmatized_words)
    return ' '.join(lemmatized_words)


# Function to detect language
def detect_language(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False


df = pd.read_csv("data.csv")
df = df[df['Comments'].str.strip().astype(bool)]

df.dropna(inplace=True)

print(df.shape)
# Resetting the index after removing rows
df.reset_index(drop=True, inplace=True)

df['IsEnglish'] = df['Comments'].apply(detect_language)
df = df[df['IsEnglish']]  # Keep only rows where 'IsEnglish' column is True

df.drop(columns=['IsEnglish'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Merge multiple lines within comments into a single line
df['Comments'] = df['Comments'].str.replace('\n', ' ')

df.reset_index(drop=True, inplace=True)

df.drop("Unnamed: 0" , axis  = 1 , inplace = True)

df['Comments'] = df['Comments'].apply(preprocess)
df = df[~df['Comments'].str.contains('amazons amk33x')]

df.reset_index(drop=True, inplace=True)

df.to_csv("cleaned.csv")

print(df.head())
print(df.shape)
