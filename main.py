import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from readability import Readability
from bertopic import BERTopic
import re

def find_tfidf(basedata):
    tfidf_vectorizer = TfidfVectorizer(input="content")
    tfidf_matrix = tfidf_vectorizer.fit_transform(basedata['summary'])

    # Get the feature names (words) from the vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get the top 10 words and their TF-IDF values for each cell
    top_words_dict = {}

    for i, row in enumerate(tfidf_matrix):
        # Get the indices of the top 10 TF-IDF values
        top_indices = row.indices[np.argsort(row.data)[-10:]]
        # Extract the corresponding words and values
        top_words = [feature_names[j] for j in top_indices]
        top_values = row.data[np.argsort(row.data)[-10:]]
        # Create a dictionary for the current cell
        cell_dict = dict(sorted(zip(top_words, top_values), key=lambda x: x[1], reverse=True))
        # Store the dictionary in the main dictionary
        top_words_dict[f'Cell_{i + 1}'] = cell_dict

    basedata['top_10_tfidf'] = list(top_words_dict.values())
    return basedata


def postag(data_to_tag):
    # two new columns to hold the new info
    data_to_tag["postagged"] = pd.Series(dtype='object')
    data_to_tag["tagcounts"] = pd.Series(dtype='object')
    for i, row in data_to_tag.iterrows():
        # tokenize text and pos tag the words
        basetext = data_to_tag.at[i, "summary"]
        tokens = nltk.word_tokenize(basetext)
        pos_tags = nltk.pos_tag(tokens)
        count_tags = {}
        # to count the pos tags
        for p in pos_tags:
            if p[1] in count_tags.keys():
                count_tags[p[1]] += 1
            else:
                count_tags[p[1]] = 1
        data_to_tag.at[i, "postagged"] = pos_tags
        data_to_tag.at[i, "tagcounts"] = count_tags

    return data_to_tag


def find_url_bias(note_data, bias_url):
    pattern = r'\bhttps:\/\/(?:www\.)?([^\/]+)\b'
    note_data["biasscore"] = pd.Series(0)
    note_data["factualityscore"] = pd.Series(0)
    note_data["biasscore"].values[:] = 0
    note_data["factualityscore"].values[:] = 0
    for index, value in note_data.iterrows():
        summary = note_data.at[index, "summary_original"]
        if "https" in summary:
            matches = re.findall(pattern, summary)
            bias_avg = []
            factuality_avg = []
            for i, k in bias_url.iterrows():
                for url in bias_url.at[i, "url"]:
                    if url in matches:
                        bias_avg.append(int(bias_url.at[i, "bias_final"]))
                        factuality_avg.append(int(bias_url.at[i, "factuality"]))
            # check if bias_avg is empty or contains both positive and neg values
            if len(bias_avg) > 0:
                if all(x >= 0 for x in bias_avg) or all(x <= 0 for x in bias_avg):
                    note_data.at[index, "biasscore"] += np.mean(bias_avg)
                    note_data.at[index, "factualityscore"] += np.mean(factuality_avg)
            else:
                note_data.drop(index, inplace=True)
    return note_data


def keep_only_relevant(data_to_check, relevant_links):
    pattern = r'\bhttps:\/\/(?:www\.)?([^\/]+)\b'
    relevant = False
    for index, value in data_to_check.iterrows():
        summary = data_to_check.at[index, "summary"]

        if "https" in summary and not relevant:
            matches = re.findall(pattern, summary)
            for i, k in relevant_links.iterrows():
                for url in relevant_links.at[i, "url"]:
                    if url in matches:
                        relevant = True
                        break
        if relevant:
            relevant = False
        else:
            data_to_check.drop(index, inplace=True)
    return data_to_check

#first step to run on the data
#removes and counts stopwords, numbers, non english text
def pre_process(data):
    stops = set(stopwords.words())
    data["stopcount"] = pd.Series(dtype='object')
    stemmer = EnglishStemmer()
    data["summary_original"] = pd.Series(dtype='object')
    for index, row in data.iterrows():
        # try to detect the language and if it isnt english drop the row
        summary = data.at[index, "summary"]
        data.at[index, "summary_original"] = summary
        try:
            lang = detect(summary)
            if lang != "en":
                data.drop(index, inplace=True)
                continue
        except:
            data.drop(index, inplace=True)
            continue
        count_stops = 0
        summary = re.sub(r'\d+', '', summary)
        stemmed_summary = [stemmer.stem(word) for word in summary.split(" ")]
        # replace the word that occur 2+ times with the stemmed version
        summary = " ".join([word if stemmed_summary.count(stemmer.stem(word)) == 1 else stemmer.stem(word) for word in
                            summary.split(" ")])
        summary = summary.split(" ")
        for word in summary:
            if word in stops:
                count_stops += 1
                summary.remove(word)
            if word.isnumeric():
                summary.remove(word)
        data.at[index, "summary"] = " ".join(summary)
        data.at[index, "stopcount"] = count_stops
    return data


# remocves everything after the first http
def remove_urls(data):
    for index, row in data.iterrows():
        summary = data.at[index, "summary"]
        if "http" in summary:
            pattern = r'\bhttp\S*\b'
            summary = re.sub(pattern, "", summary)
            data.at[index, "summary"] = summary
    return data

def check_readability(data_for_readability):
    data_for_readability["flesch_reading"] = pd.Series(dtype='object')
    data_for_readability["dale_chall"] = pd.Series(dtype='object')
    data_for_readability["gunning_fog"] = pd.Series(dtype='object')
    data_for_readability["coleman_liau"] = pd.Series(dtype='object')
    for index, row in data_for_readability.iterrows():
        r = Readability(data_for_readability.at[index, "summary_original"])
        data_for_readability.at[index, "flesch_reading"] = r.flesch()
        data_for_readability.at[index, "dale_chall"] = r.dale_chall()
        data_for_readability.at[index, "gunning_fog"] = r.gunning_fog()
        data_for_readability.at[index, "coleman_liau"] = r.coleman_liau()
    return data_for_readability


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


# refer to: https://stackoverflow.com/questions/38263039/sentiwordnet-scoring-with-python
def high_pos_neg(data_to_evaluate):
    data_to_evaluate["sentiment"] = pd.Series(dtype="object")
    data_to_evaluate["pos"] = pd.Series(dtype="object")
    data_to_evaluate["neg"] = pd.Series(dtype="object")
    data_to_evaluate["neutral"] = pd.Series(dtype="object")
    ps = PorterStemmer()

    for index, row in data_to_evaluate.iterrows():
        pos_tagged_summary = data_to_evaluate.at[index, "postagged"]
        senti_val = {x: get_sentiment(x, y) for (x, y) in pos_tagged_summary}
        senti_val = {k: v for k, v in senti_val.items() if v}
        # find the number of words that are positive, negative and neutral
        pos = 0
        neg = 0
        neutral = 0
        if senti_val:
            for k, v in senti_val.items():
                if v[0] > v[1]:
                    pos += 1
                elif v[0] < v[1]:
                    neg += 1
                else:
                    neutral += 1
        data_to_evaluate.at[index, "sentiment"] = senti_val
        data_to_evaluate.at[index, "pos"] = pos
        data_to_evaluate.at[index, "neg"] = neg
        data_to_evaluate.at[index, "neutral"] = neutral
        if neutral == 0:
            data_to_evaluate.at[index, "polarity"] = 1
        else:
            data_to_evaluate.at[index, "polarity"] = (pos + neg) / (neutral+pos+neg)

    return data_to_evaluate


def get_sentiment(word, tag):
    lemmatizer = WordNetLemmatizer()
    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.ADJ, wn.ADV):
        return []

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    return [swn_synset.pos_score(), swn_synset.neg_score()]


def sentence_sentiment(data):
    analyzer = SentimentIntensityAnalyzer()
    data["pos_sentence"] = pd.Series(dtype="object")
    data["neg_sentence"] = pd.Series(dtype="object")
    data["neutral_sentence"] = pd.Series(dtype="object")
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for index, row in data.iterrows():
        sentences = sent_detector.tokenize(data.at[index, "summary_original"].strip())
        sentiment = [analyzer.polarity_scores(sentence) for sentence in sentences]
        pos = 0
        neg = 0
        neutral = 0
        for i in sentiment:
            if i["compound"] > 0.05:
                pos += 1
            elif i["compound"] < -0.05:
                neg += 1
            else:
                neutral += 1
        data.at[index, "pos_sentence"] = pos
        data.at[index, "neg_sentence"] = neg
        data.at[index, "neutral_sentence"] = neutral
    return data

def remove_duplicates(data_to_search):
    data_to_search = data_to_search.drop_duplicates(subset="summary_original", keep="first")
    return data_to_search


def bertopic_model(data):
    model = BERTopic()
    topics, probs = model.fit_transform(data)
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    model.save("bert_models_5000", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

def remove_from_first(data_to_search, data_to_keep):
    data_to_search = data_to_search[data_to_search["noteId"].isin(data_to_keep["noteId"])]
    return data_to_search

if __name__ == '__main__':
    print("started")
    base = pd.read_csv("notes-00000.tsv", sep="\t")
    print("done_main")
