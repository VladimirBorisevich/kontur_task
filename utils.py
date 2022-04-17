import pandas as pd
import numpy as np
import re
import string
from string import punctuation
from nltk.stem.snowball import SnowballStemmer 
from nltk.corpus import stopwords
import pymorphy2
from navec import Navec

morph = pymorphy2.MorphAnalyzer()
wordnet_stemmer = SnowballStemmer("russian")
country = pd.read_csv("country.csv")
country["value"] = country["value"].apply(lambda x:x.lower())
countries = country["value"].to_list() + ["сша","туркмения","америка"]
set_of_countries = set(countries)
politics = ["алиев","пашинян","бенет","токаев","цзиньпин","ын","асад","эрдоган","макрон","тихоновская","соловьев","соловьёв",
                    "лукашенко","франциск","елизавета","джонсон","орбан","шольц","драги","санду","дуда",
                    "путин","мишустин","вучич","зеленский","нийнистё","земан","байден","навальный"]
set_of_politics = set(politics)
navec_path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(navec_path)
stopwords_rus = stopwords.words("russian")


def cnt_upper_words(row):
    words = row.split()
    counter = 0
    amount_upper_char = 0
    for word in words:
        if len(word)==1:
            continue
        elif word.isupper():
            amount_upper_char += len(word)
            counter += 1
    return counter, amount_upper_char


def prep_quotas(row):
    quoted_words = re.findall(r'«(.*?)»',row)
    cnt_chars_in_quote = 0
    if not quoted_words:
        return 0, 0
    else:
        cnt_chars_in_quote = sum([len(quota) for quota in quoted_words])
        
    return len(quoted_words), cnt_chars_in_quote


def has_quota_clause(row):
    if re.findall(r'«(.*?)»',row):
        return 1
    return 0 


def count_punc(row):
    return len(re.sub(r'[^{}]+'.format(punctuation),'',row))


def remove_stopwords(row):    
    output= [i for i in row if i not in stopwords_rus]
    return output


def stemmer(row):
    stem_text = [wordnet_stemmer.stem(word) for word in row]
    return stem_text

def lemmer(row):
    lemm_text = [morph.parse(word)[0].normalized.word for word in row]
    return lemm_text

def has_country(row):
    set_of_words = set(row)
    intr = set_of_words.intersection(set_of_countries)
    if intr:
        return len(intr)
    return 0

def politics(row):
    set_of_words = set(row)
    intr = set_of_words.intersection(set_of_politics)
    if intr:
        return len(intr)
    return 0

def has_person(row):
    for word in row:
        if morph.parse(word)[0].tag.animacy == "anim":
            return 1
    return 0 

def tense(row):
    for word in row:
        word_tense = morph.parse(word)[0].tag.tense
        if word_tense == "pres":
            return 1
        elif word_tense == "past":
            return 2
        elif word_tense == "futr":
            return 3
    return 0

def cnt_verbs(row):
    counter = 0
    for word in row:
        word_pos = morph.parse(word)[0].tag.POS
        if word_pos in {"VERB",'INFN'}:
            counter += 1
    return counter
        
def word_to_norm_to_vector(row):
    words = row.split()   
    vect = np.zeros((len(words),300))
    for i,word in enumerate(words):
        normalized = morph.parse(word)[0].normalized.word
        if normalized not in navec:
            vect[i] = navec["<pad>"]
        else:
            vect[i] = navec[normalized]
    return vect.mean(axis=0)