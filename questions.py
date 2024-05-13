import itertools
import statistics
import nltk.help
import pandas as pd
from functools import reduce
import numpy as np
import matplotlib
from matplotlib.colors import Normalize
from collections import defaultdict
from datetime import datetime, timedelta
from nltk import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from numpy import datetime64
import plotly.graph_objects as go
from main import get_sentiment
from bertopic import BERTopic
from keybert import KeyBERT
import time
from transformers import pipeline
from scipy.stats import norm, stats
import scipy.stats as st
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ast import literal_eval
import re
from main import find_tfidf, high_pos_neg, postag
import sources
from blume.table import table
import ciso8601


def two_questions(data):
    data_grouped = data.groupby("biasscore")
    data_grouped = data_grouped.agg({"pos": ["sum"], "neutral": ["sum"], "neg": ["sum"]})
    data_grouped["pos_neg_ratio"] = data_grouped["pos"] / data_grouped["neg"]
    html_table = data_grouped.to_html()
    with open('grouped.html', 'w', encoding='utf-8') as f:
        f.write(html_table)
    data_grouped = data.groupby("factualityscore")
    data_grouped = data_grouped.agg({"pos_sentence": ["sum"], "neutral_sentence": ["sum"], "neg_sentence": ["sum"]})
    data_grouped["pos_neg_ratio"] = data_grouped["pos_sentence"] / data_grouped["neg_sentence"]
    html_table = data_grouped.to_html()
    with open('grouped2.html', 'w', encoding='utf-8') as f:
        f.write(html_table)


def boxplot_column(data, column, headline, minlen=20, show_outliers=True, group_by_column="biasscore"):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)
    grouped_data = data.groupby(group_by_column)[column].agg(list)
    grouped_data = grouped_data[grouped_data.apply(lambda x: len(x) > minlen)]
    grouped_data_info = pd.DataFrame(
        {"frequency": grouped_data.apply(lambda x: len(x)), "mean": grouped_data.apply(lambda x: np.mean(x)),
         "std": grouped_data.apply(lambda x: np.std(x))})
    grouped_data_info.index = grouped_data_info.index.map(lambda x: round(x, 2))
    grouped_data_info["mean"] = grouped_data_info["mean"].map(lambda x: round(x, 2))
    grouped_data_info["std"] = grouped_data_info["std"].map(lambda x: round(x, 2))
    grouped_data.index = grouped_data.index.map(lambda x: round(x, 2))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [2, 1]})
    ax1.boxplot(grouped_data, showfliers=show_outliers)
    ax1.set_xticks(range(1, len(grouped_data.index) + 1))
    ax1.set_xticklabels(grouped_data.index)
    numeric_values = grouped_data_info.values
    norms = [Normalize() for _ in range(numeric_values.shape[1])]
    # Normalize the numeric values for coloring in each column
    normed_values = np.column_stack([norm(val) for norm, val in zip(norms, numeric_values.T)])

    # Create a table from grouped_data_info and color cells based on values
    cell_colors = plt.cm.RdYlGn(normed_values)
    table = ax2.table(cellText=grouped_data_info.values, colLabels=grouped_data_info.columns,
                      rowLabels=grouped_data_info.index, loc='center', cellLoc='center',
                      cellColours=cell_colors, fontsize=10)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    ax2.axis('off')
    ax1.set_title(headline)
    plt.tight_layout()
    plt.show()


def stacked_bar(data, columns, minlen=20, group_by_column="biasscore"):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)
    all = columns.append(group_by_column)
    data["biasscore"] = data[["biasscore"]].applymap(lambda x: round(x, 2))
    grouped_data = data[data[["biasscore"]].groupby(group_by_column).transform('size') > minlen]
    grouped_data = grouped_data[["biasscore", "neutral_sentence", "neg_sentence", "pos_sentence"]].groupby(
        group_by_column)
    grouped_data = grouped_data.mean()
    plt.figure()
    ax = grouped_data.div(grouped_data.sum(1), axis=0).plot(kind='bar', stacked=True)
    plt.legend(["neutral_sentence", "neg_sentence", "pos_sentence"], loc="upper left")
    plt.title("Sentiment distribution by biasscore")
    plt.show()

def top5_tfidf_by_posandneg(data, minfreq=1):
    data_pos = data[data["pos"] > data["neg"]]
    data_neg = data[data["pos"] < data["neg"]]
    all_tf_pos = {}
    all_tf_neg = {}
    agg_dict_pos = defaultdict(list)
    agg_dict_neg = defaultdict(list)
    for d in data_pos['top_10_tfidf']:
        for key, value in d.items():
            agg_dict_pos[key].append(value)
    for d in data_neg['top_10_tfidf']:
        for key, value in d.items():
            agg_dict_neg[key].append(value)

    average_dict_pos = {key: [sum(values) / len(values), len(values)] for key, values in agg_dict_pos.items()}
    average_dict_neg = {key: [sum(values) / len(values), len(values)] for key, values in agg_dict_neg.items()}
    df_pos = pd.DataFrame.from_dict(average_dict_pos, orient='index', columns=["average", "frequency"])
    df_neg = pd.DataFrame.from_dict(average_dict_neg, orient='index', columns=["average", "frequency"])
    df_pos = df_pos.sort_values(by=["average"], ascending=False)
    df_neg = df_neg.sort_values(by=["average"], ascending=False)
    if minfreq > 1:
        df_pos = df_pos[df_pos["frequency"] > minfreq]
        df_neg = df_neg[df_neg["frequency"] > minfreq]
    df_pos = df_pos.head(10)
    df_neg = df_neg.head(10)
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams.update({'font.size': 15})
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("pos-words")
    ax2.set_title("neg-words")
    ax1.axis('off')
    ax1.axis('tight')
    ax1.table(cellText=df_pos.values, colLabels=df_pos.columns, rowLabels=df_pos.index, loc='center', cellLoc='center')
    ax2.axis('off')
    ax2.axis('tight')
    ax2.table(cellText=df_neg.values, colLabels=df_neg.columns, rowLabels=df_neg.index, loc='center', cellLoc='center')
    plt.show()


def more_adjectives(data):
    left_leaning = data[data["bias"] == "left"]
    right_leaning = data[data["bias"] == "right"]
    neutral = data[data["bias"] == "neutral"]
    left_pos_tagged = left_leaning["postagged"]
    right_pos_tagged = right_leaning["postagged"]
    neutral_pos_tagged = neutral["postagged"]
    left_pos_tagged = left_pos_tagged.apply(lambda x: [t for t in x if t[1].startswith("J")])
    right_pos_tagged = right_pos_tagged.apply(lambda x: [t for t in x if t[1].startswith("J")])
    neutral_pos_tagged = neutral_pos_tagged.apply(lambda x: [t for t in x if t[1].startswith("J")])
    left_pos_tagged = left_pos_tagged.reset_index(drop=True)
    right_pos_tagged = right_pos_tagged.reset_index(drop=True)
    neutral_pos_tagged = neutral_pos_tagged.reset_index(drop=True)
    for i, inner_list in enumerate(left_pos_tagged):
        left_pos_tagged[i] = [
            ('pos' if sentiment[0] > sentiment[1] else 'neg' if sentiment[1] > sentiment[0] else 'neutral')
            for word, sentiment in [(word, get_sentiment(word, pos)) for word, pos in inner_list]
            if sentiment]
    for i, inner_list in enumerate(right_pos_tagged):
        right_pos_tagged[i] = [
            ('pos' if sentiment[0] > sentiment[1] else 'neg' if sentiment[1] > sentiment[0] else 'neutral')
            for word, sentiment in [(word, get_sentiment(word, pos)) for word, pos in inner_list]
            if sentiment]
    for i, inner_list in enumerate(neutral_pos_tagged):
        neutral_pos_tagged[i] = [
            ('pos' if sentiment[0] > sentiment[1] else 'neg' if sentiment[1] > sentiment[0] else 'neutral')
            for word, sentiment in [(word, get_sentiment(word, pos)) for word, pos in inner_list]
            if sentiment]
    pd.to_pickle(left_pos_tagged, "dataframes/left_adjectives.pkl")
    pd.to_pickle(right_pos_tagged, "dataframes/right_adjectives.pkl")
    pd.to_pickle(neutral_pos_tagged, "dataframes/neutral_adjectives.pkl")
    return left_pos_tagged, right_pos_tagged, neutral_pos_tagged


def adjective_graph(left, right):
    plt.figure(figsize=(10, 10))

    def average_sentiment(data):
        pos = 0
        neg = 0
        neutral = 0
        for j in data:
            for i in j:
                if i == "pos":
                    pos += 1
                elif i == "neg":
                    neg += 1
                elif i == "neutral":
                    neutral += 1
        return pos / len(data), neg / len(data), neutral / len(data), (pos / len(data) + neg / len(data)) / (
                neutral / len(data))

    left_avg = average_sentiment(left)
    right_avg = average_sentiment(right)
    plt.bar(np.arange(4), left_avg, width=0.4, label="left-leaning")
    plt.bar(np.arange(4) + 0.4, right_avg, width=0.4, label="right-leaning")
    plt.xticks(np.arange(4) + 0.4 / 2, ("positive", "negative", "neutral", "polarity"))
    plt.ylabel("average per note")
    plt.legend()
    plt.show()


def adjectives_relation(left, right, neutral):
    plt.rcParams.update(
        {'font.size': 16, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})
    pos_left = 0
    neg_left = 0
    neutral_left = 0
    pos_right = 0
    neg_right = 0
    neutral_right = 0
    pos_neutral = 0
    neg_neutral = 0
    neutral_neutral = 0
    for i in left:
        for j in i:
            if j == "pos":
                pos_left += 1
            elif j == "neg":
                neg_left += 1
            elif j == "neutral":
                neutral_left += 1
    for i in right:
        for j in i:
            if j == "pos":
                pos_right += 1
            elif j == "neg":
                neg_right += 1
            elif j == "neutral":
                neutral_right += 1
    for i in neutral:
        for j in i:
            if j == "pos":
                pos_neutral += 1
            elif j == "neg":
                neg_neutral += 1
            elif j == "neutral":
                neutral_neutral += 1
    count_left = pos_left + neg_left + neutral_left
    count_right = pos_right + neg_right + neutral_right
    count_neutral = pos_neutral + neg_neutral + neutral_neutral
    pos_left = pos_left / count_left
    neg_left = neg_left / count_left
    neutral_left = neutral_left / count_left
    pos_right = pos_right / count_right
    neg_right = neg_right / count_right
    neutral_right = neutral_right / count_right
    pos_neutral = pos_neutral / count_neutral
    neg_neutral = neg_neutral / count_neutral
    neutral_neutral = neutral_neutral / count_neutral
    print("bias", "positive", "negative", "neutral")
    print("left", pos_left, neg_left, neutral_left, count_left)
    print("right", pos_right, neg_right, neutral_right, count_right)
    print("neutral", pos_neutral, neg_neutral, neutral_neutral, count_neutral)

    plot1 = plt.bar(np.arange(3), [pos_left, neg_left, neutral_left], width=0.3, label="Left-Leaning", color="blue")
    plot2 = plt.bar(np.arange(3) + 0.3, [pos_right, neg_right, neutral_right], width=0.3, label="Right-Leaning",
                    color="orange")
    plot3 = plt.bar(np.arange(3) + 0.6, [pos_neutral, neg_neutral, neutral_neutral], width=0.3, label="Centre",
                    color="green")
    plt.xticks(np.arange(3) + 0.3, ("Positive", "Negative", "Neutral"))
    plt.ylabel("Relative frequency of adjectives")
    plt.legend()
    plt.show()


def find_url_bias(note_data, bias_url):
    pattern = r'\bhttps:\/\/(?:www\.)?([^\/]+)\b'
    sources_count = {}
    for index, value in note_data.iterrows():
        summary = note_data.at[index, "summary_original"]
        if "https" in summary:
            matches = re.findall(pattern, summary)
            bias_avg = []
            factuality_avg = []
            for i, k in bias_url.iterrows():
                if bias_url.at[i, "url"] in matches:
                    if sources_count.get(bias_url.at[i, "url"]) is None:
                        sources_count[bias_url.at[i, "url"]] = 1
                    else:
                        sources_count[bias_url.at[i, "url"]] += 1
    print(sources_count)


def add_column(data, data_to_reference, column_name):
    merged_df = pd.merge(data, data_to_reference, on="noteId")
    merged_df = merged_df.drop_duplicates(subset="noteId")
    merged_df = merged_df.reset_index(drop=True)
    data[column_name] = merged_df[column_name]
    return data


def polarization_trends_faulty(data, count=10):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)
    left = data[data["bias"] == "left"]
    right = data[data["bias"] == "right"]
    left = left.filter(["dateTime", "polarity"])
    right = right.filter(["dateTime", "polarity"])
    left['dateTime_ordinal'] = left['dateTime'].apply(lambda p: p.toordinal())
    right['dateTime_ordinal'] = right['dateTime'].apply(lambda p: p.toordinal())
    grouped_left = left.groupby(pd.Grouper(key='dateTime', freq='W-Mon'))
    grouped_right = right.groupby(pd.Grouper(key='dateTime', freq='W-Mon'))
    grouped_left = grouped_left.mean()
    grouped_right = grouped_right.mean()
    grouped_left["count"] = left.groupby(pd.Grouper(key='dateTime', freq='W-Mon'))["dateTime"].count()
    grouped_right["count"] = right.groupby(pd.Grouper(key='dateTime', freq='W-Mon'))["dateTime"].count()
    grouped_left = grouped_left[grouped_left["count"] >= count]
    grouped_right = grouped_right[grouped_right["count"] >= count]
    grouped_left = grouped_left.reset_index()
    grouped_right = grouped_right.reset_index()

    b_left, a_left = np.polyfit(grouped_left['dateTime_ordinal'], grouped_left['polarity'], 1)
    b_right, a_right = np.polyfit(grouped_right['dateTime_ordinal'], grouped_right['polarity'], 1)
    print(grouped_left["dateTime_ordinal"])
    xseq_left = np.linspace(grouped_left['dateTime_ordinal'].min(), grouped_left['dateTime_ordinal'].max(), 146)
    xseq_right = np.linspace(grouped_right['dateTime_ordinal'].min(), grouped_right['dateTime_ordinal'].max(), 58)
    fig, ax = plt.subplots(figsize=(10, 10))

    left_reg, = ax.plot(grouped_left["dateTime"], b_left * xseq_left + a_left, color="blue",
                        label="Left-leaning linear regression")
    right_reg, = ax.plot(grouped_right["dateTime"], b_right * xseq_right + a_right, color="orange",
                         label="Right-leaning linear regression")

    left_largest = grouped_left.nlargest(5, 'polarity')
    right_largest = grouped_right.nlargest(5, "polarity")

    ax.scatter(grouped_left["dateTime"], grouped_left['polarity'], label="left-leaning")
    ax.scatter(grouped_right["dateTime"], grouped_right['polarity'], label="right-leaning")

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Polarity')
    for x, y, count in zip(left_largest["dateTime"], left_largest['polarity'], left_largest['count']):
        formatted_date = x.strftime("%Y-%m-%d")
        ax.annotate(str(count) + " " + formatted_date, xy=(x, y + 0.01), color="blue", ha="center",
                    va="center", fontsize="large")
    for x, y, count in zip(right_largest["dateTime"], right_largest['polarity'], right_largest['count']):
        formatted_date = x.strftime("%Y-%m-%d")
        ax.annotate(str(count) + " " + formatted_date, xy=(x, y + 0.01), color="orange", ha="center", va="center",
                    fontsize="large")
    plt.ion()
    plt.show()


def polarization_trends(data, count=10):
    plt.rcParams.update(
        {'font.size': 16, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})

    left = data[data["bias"] == "left"]
    right = data[data["bias"] == "right"]
    left = left.filter(["dateTime", "polarity"])
    right = right.filter(["dateTime", "polarity"])
    left['dateTime_ordinal'] = left['dateTime'].apply(lambda p: p.toordinal())
    right['dateTime_ordinal'] = right['dateTime'].apply(lambda p: p.toordinal())
    grouped_left = left.groupby(pd.Grouper(key='dateTime', freq='W-Mon'))
    grouped_right = right.groupby(pd.Grouper(key='dateTime', freq='W-Mon'))
    grouped_left = grouped_left.mean()
    grouped_right = grouped_right.mean()
    grouped_left["count"] = left.groupby(pd.Grouper(key='dateTime', freq='W-Mon'))["dateTime"].count()
    grouped_right["count"] = right.groupby(pd.Grouper(key='dateTime', freq='W-Mon'))["dateTime"].count()
    grouped_left = grouped_left.reset_index()
    grouped_right = grouped_right.reset_index()
    grouped_left = grouped_left[grouped_left["count"] >= count]
    grouped_right = grouped_right[grouped_right["count"] >= count]

    b_left, a_left = np.polyfit(grouped_left['dateTime_ordinal'], grouped_left['polarity'], 1)
    b_right, a_right = np.polyfit(grouped_right['dateTime_ordinal'], grouped_right['polarity'], 1)
    xseq_left = np.linspace(grouped_left['dateTime_ordinal'].min(), grouped_left['dateTime_ordinal'].max(), 146)
    xseq_right = np.linspace(grouped_right['dateTime_ordinal'].min(), grouped_right['dateTime_ordinal'].max(), 58)
    fig, ax = plt.subplots(figsize=(10, 10))

    left_reg, = ax.plot(grouped_left["dateTime"], b_left * xseq_left + a_left, color="blue",
                        label="Left-leaning linear regression")
    right_reg, = ax.plot(grouped_right["dateTime"], b_right * xseq_right + a_right, color="orange",
                         label="Right-leaning linear regression")

    left_largest = grouped_left.nlargest(5, 'polarity')
    right_largest = grouped_right.nlargest(5, "polarity")


    ax.scatter(grouped_left["dateTime"], grouped_left['polarity'], label="Left-leaning")
    ax.scatter(grouped_right["dateTime"], grouped_right['polarity'], label="Right-leaning")

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Polarity')
    print(left_largest)
    for x, y, count in zip(left_largest["dateTime"], left_largest['polarity'], left_largest['count']):
        formatted_date = x.strftime("%Y-%m-%d")
        ax.annotate(str(count) + " " + formatted_date, xy=(x, y + 0.01), color="blue", ha="center",
                    va="center", fontsize="large")
    for x, y, count in zip(right_largest["dateTime"], right_largest['polarity'], right_largest['count']):
        formatted_date = x.strftime("%Y-%m-%d")
        ax.annotate(str(count) + " " + formatted_date, xy=(x, y + 0.01), color="orange", ha="center", va="center",
                    fontsize="large")
    plt.ion()
    plt.show()



def sentiment_by_leaning(data, variant):
    left = data[data["biasscore"] > 0]
    right = data[data["biasscore"] < 0]
    left_pos = left["pos_sentence"].sum()
    left_neg = left["neg_sentence"].sum()
    left_neutral = left["neutral_sentence"].sum()
    right_pos = right["pos_sentence"].sum()
    right_neg = right["neg_sentence"].sum()
    right_neutral = right["neutral_sentence"].sum()
    if variant == 1:
        left_total = left_pos + left_neg + left_neutral
        right_total = right_pos + right_neg + right_neutral
        left_pos = left_pos / left_total
        left_neg = left_neg / left_total
        left_neutral = left_neutral / left_total
        right_pos = right_pos / right_total
        right_neg = right_neg / right_total
        right_neutral = right_neutral / right_total
    if variant == 2:
        left_pos = left_pos / len(left)
        left_neg = left_neg / len(left)
        left_neutral = left_neutral / len(left)
        right_pos = right_pos / len(right)
        right_neg = right_neg / len(right)
        right_neutral = right_neutral / len(right)
    plt.bar(np.arange(3), [left_pos, left_neg, left_neutral], width=0.4, label="left-leaning")
    plt.bar(np.arange(3) + 0.4, [right_pos, right_neg, right_neutral], width=0.4, label="right-leaning")
    plt.xticks(np.arange(3) + 0.4 / 2, ("positive", "negative", "neutral"))

    plt.legend()
    if variant == 1:
        plt.ylabel("percentage")
        plt.title("Odds of a sentence being positive, negative, or neutral if it is in a left or right leaning note")
    if variant == 2:
        plt.ylabel("average amount")
        plt.title("Avg amount of positive, negative, or neutral sentences in a left or right leaning note")
    plt.show()


def berting(topic_model):
    docs = pd.read_pickle("dataframes/base_for_berting_shortened_5000.pkl")
    docs = docs["summary_original"].tolist()
    topic_model.get_document_info(docs).to_csv("csvs/berting2.csv")


def bert_graph(bertdf, base):
    base = base["biasscore"]
    bert_data = pd.concat([bertdf, base], axis=1)
    bert_data['biasscore'] = np.where(bert_data['biasscore'] > 0, 'left',
                                      np.where(bert_data['biasscore'] < 0, 'right', bert_data['biasscore']))
    bert_data.to_csv("csvs/berting_bias.csv")


def bert_graph2(bertdf):
    bertdf = bertdf.query('biasscore != "0.0" and Topic != -1')
    topic_pairs = bertdf[["Topic", "Representation"]].drop_duplicates()

    selected_topics = [0, 1, 2, 3, 4]
    bertdf_selected = bertdf[bertdf['Topic'].isin(selected_topics)]
    topic_pairs_selected = topic_pairs[topic_pairs['Topic'].isin(selected_topics)].sort_values(by=['Topic'])
    topic_counts = bertdf_selected.groupby(['Topic', 'biasscore']).size().reset_index(name='Count')
    topic_counts_pivot = topic_counts.pivot(index='Topic', columns='biasscore', values='Count').reset_index()


    fig, (ax, ax_table) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    bar_width = 0.35
    left_positions = topic_counts_pivot['Topic'] - bar_width / 2
    right_positions = topic_counts_pivot['Topic'] + bar_width / 2

    ax.bar(left_positions, topic_counts_pivot['left'], width=bar_width, label='left')
    ax.bar(right_positions, topic_counts_pivot['right'], width=bar_width, label='right')

    ax.set_xlabel('Topic')
    ax.set_ylabel('Group Size')
    ax.legend()

    table = ax_table.table(cellText=topic_pairs_selected.values, colLabels=topic_pairs_selected.columns, loc='center',
                           cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax_table.axis('off')

    plt.show()


def keyberting(notesdf, topic_nr, top=20, ngram=2):
    left_leaning = notesdf.query('biasscore == "left" and Topic == @topic_nr')
    right_leaning = notesdf.query('biasscore == "right" and Topic == @topic_nr')
    left_leaning = left_leaning["Document"].tolist()
    amount_left = len(left_leaning)
    right_leaning = right_leaning["Document"].tolist()
    amount_right = len(right_leaning)
    kw_model = KeyBERT()
    for i, note in enumerate(left_leaning):
        # Remove everything after the first occurence of http
        note = re.sub(r"http\S+", "", note)
        left_leaning[i] = note
    for i, note in enumerate(right_leaning):
        note = re.sub(r"http\S+", "", note)
        right_leaning[i] = note
    # merge list into one string
    left_leaning = " ".join(left_leaning)
    right_leaning = " ".join(right_leaning)
    left_keywords = kw_model.extract_keywords(left_leaning, top_n=top, keyphrase_ngram_range=(ngram, ngram))
    right_keywords = kw_model.extract_keywords(right_leaning, top_n=top, keyphrase_ngram_range=(ngram, ngram))
    left_keyword_occurences = [[keyword, score] for keyword, score in left_keywords]
    right_keyword_occurences = [[keyword, score] for keyword, score in right_keywords]

    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # table(ax, left_keyword_occurences, loc='center', cellLoc='center')
    # table(ax2, right_keyword_occurences, loc='center', cellLoc='center')
    # ax.axis('off')
    # ax2.axis('off')
    # ax.set_title("Left leaning")
    # ax2.set_title("Right leaning")
    # plt.show()
    return left_keywords, right_keywords


def keyberting_topicless(notesdf, top=20, ngram=2):
    left_leaning = notesdf[notesdf["bias"] == "left"]
    right_leaning = notesdf[notesdf["bias"] == "right"]
    left_leaning = left_leaning["summary_original"].tolist()
    amount_left = len(left_leaning)
    right_leaning = right_leaning["summary_original"].tolist()
    amount_right = len(right_leaning)
    kw_model = KeyBERT()
    for i, note in enumerate(left_leaning):
        note = re.sub(r"http\S+", "", note)
        left_leaning[i] = note
    for i, note in enumerate(right_leaning):
        note = re.sub(r"http\S+", "", note)
        right_leaning[i] = note
    # merge list into one string
    left_leaning = " ".join(left_leaning)
    right_leaning = " ".join(right_leaning)
    left_keywords = kw_model.extract_keywords(left_leaning, top_n=top, keyphrase_ngram_range=(ngram, ngram))
    right_keywords = kw_model.extract_keywords(right_leaning, top_n=top, keyphrase_ngram_range=(ngram, ngram))

    left_keyword_occurences = [[keyword, score] for keyword, score in left_keywords]
    right_keyword_occurences = [[keyword, score] for keyword, score in right_keywords]
    return left_keywords, right_keywords


def stem_combine(phrase_scores, amount=20):
    # phrase scores to dictionary from 2d array
    phrase_scores = dict(phrase_scores)
    stemmer = SnowballStemmer("english")
    highest_score_phrases = {}
    # Step 3: Iterate through each phrase and its score
    for phrase, score in phrase_scores.items():
        # Split the phrase into words and stem each word
        words = phrase.split()
        for i, word in enumerate(words):
            if word == ("ukrainian" or "ukrainians"):
                words[i] = "ukrain"
        phrase = ' '.join(words)
        stemmed_words = [stemmer.stem(word) for word in phrase.split()]
        # Sort the stemmed words to create a standardized key
        sorted_stemmed_words = tuple(sorted(stemmed_words))

        # Check if this stemmed list has been seen before
        if sorted_stemmed_words in highest_score_phrases:
            # If so, add the score to the existing score for this stemmed list
            existing_score, existing_phrase = highest_score_phrases[sorted_stemmed_words]
            new_score = existing_score + score
            # Determine which phrase should be retained (the one with the higher score)
            if score > existing_score:
                highest_score_phrases[sorted_stemmed_words] = (new_score, phrase)
            else:
                highest_score_phrases[sorted_stemmed_words] = (new_score, existing_phrase)
        else:
            # If this is a new stemmed list, add it to the dictionary
            highest_score_phrases[sorted_stemmed_words] = (score, phrase)
    # Step 4: Extract the final phrases with their total scores
    final_phrases_with_scores = {phrase: score for (score, phrase) in highest_score_phrases.values()}
    # Sorting the dictionary by score in descending order
    sorted_final_phrases = sorted(final_phrases_with_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_final_phrases


def plot_phrases(left, right):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    table1 = table(ax, left, cellLoc="center", loc="center")
    table2 = table(ax1, right, cellLoc="center", loc="center")
    ax.axis('off')
    ax1.axis("off")
    ax.set_title("Left leaning")
    ax1.set_title("Right leaning")
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    plt.show()


def great_plot_phrases(left, right):
    left_df = pd.DataFrame(left, columns=["Left-Phrase", "Left-Score"])
    right_df = pd.DataFrame(right, columns=["Right-Phrase", "Right-Score"])
    merged = pd.concat([left_df, right_df], axis=1)
    fig = go.Figure(data=[go.Table(header=dict(values=list(merged.columns)),
                                   cells=dict(
                                       values=[merged["Left-Phrase"], merged["Left-Score"], merged["Right-Phrase"],
                                               merged["Right-Score"]]))
                          ])
    fig.show()


def calculate_polarity(row):
    if row["neutral"] == 0:
        return 1
    else:
        return (row["pos"] + row["neg"]) / row["neutral"]


def run_on_all(data, topics, min_score=0.4):
    for topic in topics:
        left, right = keyberting(data, topic, 100, 2)
        vasak = pd.DataFrame(stem_combine(left), columns=["Left-Keyword", "Score"])
        parem = pd.DataFrame(stem_combine(right), columns=["Right-Keyword", "Score"])
        vasak = vasak[vasak["Score"] > min_score]
        parem = parem[parem["Score"] > min_score]
        merged = pd.concat([vasak, parem], axis=1)
        merged.to_csv(f"csvs/keywords_{topic}.csv")


# given the ending dates of a week and the basic data, return a table for a certain bias, where pos and neg words are counted
def polarity_weekly_table(data, dates):
    for date in dates:
        pos_words = {}
        neg_words = {}
        date = datetime.strptime(date, "%d/%m/%Y")
        start_of_week = date - timedelta(days=5)
        print(start_of_week)
        start_of_week = datetime.strftime(start_of_week, "%d-%m-%Y")
        filtered = data[
            (data["dateTime"].dt.date >= pd.to_datetime(start_of_week, format='%d-%m-%Y').date()) &
            (data["dateTime"].dt.date <= pd.to_datetime(date, format='%d/%m/%Y').date()) &
            (data["biasscore"] < 0)
            ]
        for note in filtered["sentiment"]:
            for key, value in note.items():
                if value[0] > value[1]:
                    if key not in pos_words:
                        pos_words[key] = 1
                    else:
                        pos_words[key] += 1
                else:
                    if key not in neg_words:
                        neg_words[key] = 1
                    else:
                        neg_words[key] += 1

        pos_words = dict(sorted(pos_words.items(), key=lambda item: item[1], reverse=True))
        neg_words = dict(sorted(neg_words.items(), key=lambda item: item[1], reverse=True))
        pos_df = pd.DataFrame(list(pos_words.items()), columns=["Pos-Word", "Pos-Count"])
        neg_df = pd.DataFrame(list(neg_words.items()), columns=["Neg-Word", "Neg-Count"])
        merged = pd.concat([pos_df, neg_df], axis=1)
        merged.to_csv(f"csvs\Right_{start_of_week}.csv")
    return


# week/bias/group size/avg/min/max/
def polarity_table(data, left_dates, right_dates):
    left = data[data["biasscore"] > 0]
    right = data[data["biasscore"] < 0]
    left_group = left.groupby(pd.Grouper(key="dateTime", freq="W-Mon"))
    right_group = right.groupby(pd.Grouper(key="dateTime", freq="W-Mon"))
    all = []
    for name, group in left_group:
        dt = datetime.fromtimestamp(name.timestamp())
        string_time = dt.strftime("%d/%m/%Y")
        if string_time in left_dates:
            mean = group["polarity"].mean()
            min = group["polarity"].min()
            max = group["polarity"].max()
            size = len(group)
            print("--------------------------------------------------------")
            print(name)
            print(group["summary_original"])
            tabel = ["Left", string_time, mean, min, max, size]
            all.append(tabel)
    for name, group in right_group:
        dt = datetime.fromtimestamp(name.timestamp())
        string_time = dt.strftime("%d/%m/%Y")
        if string_time in right_dates:
            mean = group["polarity"].mean()
            min = group["polarity"].min()
            max = group["polarity"].max()
            size = len(group)
            tabel = ["Right", string_time, mean, min, max, size]
            all.append(tabel)
    alldf = pd.DataFrame(all, columns=["Bias", "Date", "Mean", "Min", "Max", "Size"])



def filter_dates(data, date, bias):
    date = datetime.strptime(date, "%d/%m/%Y")
    start_of_week = date - timedelta(days=6)
    start_of_week = datetime.strftime(start_of_week, "%d-%m-%Y")
    filtered = data[
        (data["dateTime"].dt.date >= pd.to_datetime(start_of_week, format='%d-%m-%Y').date()) &
        (data["dateTime"].dt.date <= pd.to_datetime(date, format='%d/%m/%Y').date()) &
        (data["bias"] == bias)
        ]
    return filtered


def get_period(data, bias="left", start_date="15/09/2021", end_date="27/09/2021"):
    if bias == "left":
        filtered = data[(data["dateTime"] >= pd.to_datetime(start_date, format="%d/%m/%Y")) & (
                data["dateTime"] <= pd.to_datetime(end_date, format="%d/%m/%Y")) & data["biasscore"] > 0]
    if bias == "right":
        filtered = data[(data["dateTime"] >= pd.to_datetime(start_date, format="%d/%m/%Y")) & (
                data["dateTime"] <= pd.to_datetime(end_date, format="%d/%m/%Y"))]
        filtered = filtered[filtered["biasscore"] < 0]
    keybert = KeyBERT()
    summaries = " ".join(filtered["summary"].tolist())
    print(filtered["biasscore"])
    keywords = keybert.extract_keywords(summaries, top_n=30, keyphrase_ngram_range=(2, 2))
    keywords = stem_combine(keywords)
    keywords_df = pd.DataFrame([[keyword, score] for keyword, score in keywords])
    csv = keywords_df.to_csv("csvs/ngrams_right_15-27Y21.csv")
    print(keywords)


def note_sentiment(data_sen):
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    data_sen["original_nolinks"] = data_sen["summary_original"].apply(lambda x: re.sub(r"http\S+", "", x))
    print("column created")
    data_sen["sentiment"] = data_sen["original_nolinks"].apply(lambda x: pipe(x))
    return data_sen


def agg_groups(data):
    data["bias"] = data["biasscore"].apply(lambda x: "left" if x > 0.5 else "right" if x < -0.5 else "neutral")
    return data


def confidence_sampling(data):
    # only keep rows where the confidence is >0.7
    data = data[data["sentiment"].apply(lambda x: x[0]["score"] > 0.7)]
    return data


def stacked_bar_note(data, group_by_column="bias"):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)

    grouped_data = data[["bias", "note_sentiment"]]
    grouped_data = grouped_data.groupby(group_by_column)
    grouped_data = grouped_data["note_sentiment"].value_counts().unstack().fillna(0)
    grouped_data.columns = [x.capitalize() for x in grouped_data.columns]
    grouped_data = grouped_data[["Positive", "Neutral", "Negative"]]
    print(grouped_data)
    utilize_plot_ci(grouped_data)


def column_seperate_note(data, column):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)

    grouped_data = data[["bias", "note_sentiment"]].groupby("bias")
    grouped_data = grouped_data["note_sentiment"].value_counts().unstack().fillna(0)
    grouped_data = grouped_data[["neutral", "negative", "positive"]]
    grouped_data["part_of_whole"] = grouped_data[column] / grouped_data.sum(axis=1)
    print(grouped_data)
    ax = grouped_data["part_of_whole"].plot(kind='bar')
    plt.ylabel("Percentage of notes", fontdict=font)
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.show()


# the percentage of sentences in each sentiment category for each bias
def stacked_bar_sentence(data):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)
    data = data[["bias", "pos_sentence", "neg_sentence", "neutral_sentence"]]
    data = data.rename(columns={"pos_sentence": "Positive", "neg_sentence": "Negative", "neutral_sentence": "Neutral"})
    grouped_data = data.groupby("bias").sum()
    grouped_data = grouped_data[["Positive", "Neutral", "Negative"]]
    print(grouped_data)
    utilize_plot_ci(grouped_data)


def column_seperate(data, column):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)
    data = data[["bias", "pos_sentence", "neg_sentence", "neutral_sentence"]]
    grouped_data = data.groupby("bias").sum()
    grouped_data = grouped_data[["neutral_sentence", "neg_sentence", "pos_sentence"]]
    grouped_data["part_of_whole"] = grouped_data[column] / grouped_data.sum(axis=1)
    print(grouped_data)
    ax = grouped_data["part_of_whole"].plot(kind='bar')
    plt.ylabel("Percentage of sentences", fontdict=font)
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.show()

def column_seperate2(data, column):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)
    data["sentence_sentiment"] = data.apply(
        lambda x: "positive" if x["pos_sentence"] > x["neg_sentence"] else "negative" if x["neg_sentence"] > x[
            "pos_sentence"] else "neutral", axis=1)
    grouped_data = data[["bias", "sentence_sentiment"]].groupby("bias")
    grouped_data = grouped_data["sentence_sentiment"].value_counts().unstack().fillna(0)
    grouped_data = grouped_data[["neutral", "negative", "positive"]]
    grouped_data["part_of_whole"] = grouped_data[column] / grouped_data.sum(axis=1)
    print(grouped_data)
    ax = grouped_data["part_of_whole"].plot(kind='bar')
    plt.ylabel("Percentage of sentences", fontdict=font)
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.show()


def stacked_bar_sentence2(data, group_by_column="bias"):
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)
    data["sentence_sentiment"] = data.apply(
        lambda x: "Positive" if x["pos_sentence"] > x["neg_sentence"] else "Negative" if x["neg_sentence"] > x[
            "pos_sentence"] else "Neutral", axis=1)
    grouped_data = data[["bias", "sentence_sentiment"]].groupby(group_by_column)
    grouped_data = grouped_data["sentence_sentiment"].value_counts().unstack().fillna(0)
    grouped_data = grouped_data[["Positive", "Neutral", "Negative"]]
    print(grouped_data)
    utilize_plot_ci(grouped_data)


def top5_sentiment(data):
    left = data[data["bias"] == "left"]
    right = data[data["bias"] == "right"]
    left["score"] = left["sentiment"].apply(lambda x: x[0]["score"])
    right["score"] = right["sentiment"].apply(lambda x: x[0]["score"])
    left_pos = left[left["note_sentiment"] == "positive"].sort_values(by="score", ascending=False).head(5)
    left_neg = left[left["note_sentiment"] == "negative"].sort_values(by="score", ascending=False).head(5)
    right_pos = right[right["note_sentiment"] == "positive"].sort_values(by="score", ascending=False).head(5)
    right_neg = right[right["note_sentiment"] == "negative"].sort_values(by="score", ascending=False).head(5)
    left_pos = left_pos[["summary_original", "score"]]
    left_neg = left_neg[["summary_original", "score"]]
    right_pos = right_pos[["summary_original", "score"]]
    right_neg = right_neg[["summary_original", "score"]]
    left_pos["group"] = "left_pos"
    left_neg["group"] = "left_neg"
    right_pos["group"] = "right_pos"
    right_neg["group"] = "right_neg"
    top5 = pd.concat([left_pos, left_neg, right_pos, right_neg])
    top5.to_csv("csvs/top5.csv")


def add_note_sentiment(data):
    data["note_sentiment"] = data["sentiment"].apply(lambda x: x[0]["label"])
    return data


def add_dateTime(data):
    data["dateTime"] = pd.to_datetime(data["createdAtMillis"], unit="ms")
    return data


def add_note_sentiment_based_on_words(data):
    data["note_sentiment_words"] = data.apply(
        lambda x: "positive" if x["pos"] > x["neg"] else "negative" if x["neg"] > x["pos"] else "neutral", axis=1)
    return data


def find_standarddev(data):
    left = data[data["bias"] == "left"]
    right = data[data["bias"] == "right"]
    left = left[left["dateTime"].dt.year == 2023]
    right = right[right["dateTime"].dt.year == 2023]
    left = left.filter(["dateTime", "polarity"])
    right = right.filter(["dateTime", "polarity"])

    grouped_left = left.groupby(pd.Grouper(key='dateTime', freq='W-Mon'))
    grouped_right = right.groupby(pd.Grouper(key='dateTime', freq='W-Mon'))
    grouped_left = grouped_left.mean()
    grouped_right = grouped_right.mean()

    grouped_left["count"] = left.groupby(pd.Grouper(key='dateTime', freq='W-Mon')).size()
    grouped_right["count"] = right.groupby(pd.Grouper(key='dateTime', freq='W-Mon')).size()

    grouped_left = grouped_left[grouped_left["count"] >= 10]
    grouped_right = grouped_right[grouped_right["count"] >= 10]

    left_std = grouped_left["polarity"].std()
    right_std = grouped_right["polarity"].std()
    print("Left std: ", left_std)
    print("Right std: ", right_std)


def sentence_boxplots(data, type="dale_chall"):
    plt.rcParams.update(
        {'font.size': 16, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})
    left = data[data["bias"] == "left"]
    right = data[data["bias"] == "right"]
    neutral = data[data["bias"] == "neutral"]
    left["readability_score"] = left[type].apply(lambda x: x.score)
    right["readability_score"] = right[type].apply(lambda x: x.score)
    neutral["readability_score"] = neutral[type].apply(lambda x: x.score)
    scores = []
    scores.append(left["readability_score"])
    scores.append(neutral["readability_score"])
    scores.append(right["readability_score"])
    print("Left: ", np.quantile(scores[0], [0.25, 0.5, 0.75], axis=0))
    print("Neutral: ", np.quantile(scores[1], [0.25, 0.5, 0.75], axis=0))
    print("Right: ", np.quantile(scores[2], [0.25, 0.5, 0.75], axis=0))

    colors = ['blue', 'green', 'orange']
    plot = plt.boxplot(scores, labels=["Left-Leaning", "Centre", "Right-Leaning"], showfliers=False, patch_artist=True)
    for patch, color in zip(plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.ylabel("Readability Score")
    plt.show()


def plot_with_ci(ax, dataframe, category, color, relative_freq_df):
    freq = relative_freq_df[category]
    p = freq / 100
    n = dataframe.sum(axis=1)

    se = np.sqrt(p * (1 - p) / n)
    ci = se * norm.ppf(0.995)
    ci_lower = (p - ci) * 100
    ci_upper = (p + ci) * 100
    error = [freq - ci_lower, ci_upper - freq]

    ax.bar(freq.index, freq, width=0.7, color=color, label=category)
    ax.errorbar(freq.index, freq, yerr=error, fmt='o', color='black', capsize=5)
    ax.set_xlabel('Bias')
    ax.set_ylabel('Relative Frequency (%)')
    ax.margins(y=0.1)

    for i, value in enumerate(freq):
        ax.text(x=i, y=ci_upper[i], s=f'{dataframe.at[freq.index[i], category]}', ha='center', va='bottom')


def utilize_plot_ci(sentiment_df):
    stance_classes = ['Left-Leaning', 'Neutral', 'Right-Leaning']
    sentiment_df.index = sentiment_df.index.str.replace('left', 'Left-Leaning').str.replace('right',
                                                                                            'Right-Leaning').str.replace(
        'neutral', 'Center')
    sentiment_classes = ['Positive', 'Neutral', 'Negative']
    sentiment_df = sentiment_df.astype(int)
    relative_freq_df = sentiment_df.div(sentiment_df.sum(axis=1), axis=0) * 100

    colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}

    plt.rcParams.update(
        {'font.size': 16, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))

    relative_freq_df.plot(kind='bar', width=0.7, stacked=True, ax=axes[0],
                          color=[colors[col] for col in relative_freq_df.columns])
    axes[0].set_xlabel('Bias')
    axes[0].set_ylabel('Relative Frequency (%)')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].get_legend().remove()
    axes[0].margins(y=0.15)
    for i, (index_value, row) in enumerate(relative_freq_df.iterrows()):
        y_offset = row.sum()
        axes[0].text(x=i, y=y_offset, s=f'{sentiment_df.loc[index_value].sum()}', ha='center', va='bottom')

    plot_with_ci(axes[1], sentiment_df, 'Negative', 'red', relative_freq_df)

    plot_with_ci(axes[2], sentiment_df, 'Positive', 'green', relative_freq_df)

    fig.legend(sentiment_classes, title='Sentiment', loc='lower center', bbox_to_anchor=(0.515, 0))

    plt.tight_layout()
    plt.show()


def words_and_adjectives(all_notes):
    left = data[data["bias"] == "left"]
    right = data[data["bias"] == "right"]
    neutral = data[data["bias"] == "neutral"]
    l_w, l_a = find_total_words(left)
    r_w, r_a = find_total_words(right)
    n_w, n_a = find_total_words(neutral)
    plt.rcParams.update(
        {'font.size': 16, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})
    words_adjectives = pd.DataFrame({"Bias": ["Left-Leaning", "Centre", "Right-Leaning"],
                                     "Adjectives": [l_a, n_a, r_a],
                                     "Other Words": [l_w - l_a, n_w - n_a, r_w - r_a]})
    words_adjectives.plot(x="Bias", kind="bar", stacked=True, color=["gold", "mediumpurple"], rot=0)
    for i, (index_value, row) in enumerate(words_adjectives.iterrows()):
        plt.text(x=i, y=row["Adjectives"] / 2, s=f'{round(row["Adjectives"], 2)}', ha='center', va='bottom')
        plt.text(x=i, y=row["Adjectives"] + row["Other Words"] / 2, s=f'{round(row["Other Words"], 2)}', ha='center',
                 va='bottom')

    plt.xlabel("Bias")
    plt.ylabel("Average Number of Words")
    plt.show()


def find_total_words(bias_group):
    bias_group["original_nolinks"] = bias_group["summary_original"].apply(lambda x: re.sub(r"http\S+", "", x))
    total_words = 0
    total_adjectives = 0
    word_counts = []
    adj_counts = []
    for i, row in bias_group.iterrows():
        words = row["original_nolinks"].replace(r"[^\W\d]+", "").split(" ")
        words = [word for word in words if word]
        total_words += len(words)
        pos = row["postagged"]
        adj = 0
        for p in pos:
            if p[1] == "JJ" or p[1] == "JJR" or p[1] == "JJS":
                total_adjectives += 1
                adj += 1
        word_counts.append(len(words))
        adj_counts.append(adj)

    return total_words / len(bias_group), total_adjectives / len(bias_group)

def source_bias_graph(data):
    bias_counts = data["bias_final"].value_counts()
    bias_df = pd.DataFrame({"Bias": bias_counts.index, "Source Count": bias_counts.values}).sort_values(by="Bias", ascending=False)
    bias_df["Bias"] = bias_df["Bias"].replace({-2: "Right", -1: "Right-Center", 0: "Center", 1: "Left-Center", 2: "Left"})
    print(bias_df)
    plt.rcParams.update(
        {'font.size': 16, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})
    bias_df.plot(x="Bias", y="Source Count", kind="bar", color="dodgerblue", rot=0)
    plt.xlabel("Bias")
    plt.ylabel("Source Count")
    plt.show()

if __name__ == "__main__":
    pd.options.display.width = 0
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', 10000)
    pd.set_option('display.max_rows', 10000)
    print("Loading data")
    #data = pd.read_pickle("dataframes/note_sentiment_dateTime.pkl")
    data = pd.read_csv("csvs/urls.csv")
    source_bias_graph(data)
    print("Done")