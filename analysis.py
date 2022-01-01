import pickle

from os.path import join, split
from os import getcwd
from datetime import datetime
from collections import defaultdict
from nltk import word_tokenize

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = "annotations"
FILE = "corpus_unannotated_auto-annotated_full.csv"

all_keywords = {"ansiedler", "aussiedler", "migration", "einwanderung", "einwanderer", "zustrom"}


def main():
    plot_comparison_individuals()
    #plot_straight()
    #plot_comparison()
    pass


def plot_comparison_individuals():
    df = pd.read_csv(join(getcwd(), "annotations", "corpus_unannotated_auto-annotated_full.csv"))

    df = df[(df["Label"] == 0) | (df["Label"] == 1)]

    # year: {"word": 4782, "worders":57,...}
    dic = defaultdict(lambda: defaultdict_with_counter(1))

    # fill data

    for sentence, year, label in zip(df["sentence"], df["year"], df["Label"]):
        sentence = word_tokenize(sentence)
        for word in sentence:
            if word.lower() in all_keywords:
                dic[year][word.lower()][label] += 1

    fig, ax = plt.subplots(1, figsize=(8, 5))
    print(dic)

    for keyword in all_keywords:
        x_vals, y_vals = [], []
        for year, _ in sorted(dic.items(), key=lambda x: x[0]):
            val1, val2 = dic[year][keyword][0], dic[year][keyword][1]
            if val1 == 1 == val2:
                pass
                #y_vals.append(np.nan)
            else:
                y_vals.append(val1 / val2)
                x_vals.append(year)

        ax.set_yscale('log')
        ax.plot(x_vals, y_vals, label=keyword)

    ax.set_xlabel("Year")
    ax.set_ylabel("#Soli / #Anti sentences")

    ax.grid(True, which="both")
    ax.legend()

    plt.axhline(y=1, color='black', linestyle='-')
    plt.suptitle("Relation of (anti-)solidary sentences towards migrants")
    plt.tight_layout()
    plt.savefig(join(getcwd(), "Frequency_graphs", "auto-annotated-solidarity-relation-individuals-migrants"))
    plt.show()


def defaultdict_with_counter(depth):
    if depth == 0:
        return defaultdict(lambda: 1)
    return defaultdict(lambda: defaultdict_with_counter(depth-1))


def plot_comparison():
    df = pd.read_csv(join(PATH, FILE))
    # file: date
    with open(join("dicts", 'dicts/dates.pkl'), 'rb') as _f:
        dict_dates = pickle.load(_f)

    fig, ax = plt.subplots(1, figsize=(9, 5))

    for keyword in ["woman", "migrant"]:
        df_cat = df[df["category"] == keyword]

        df_soli = df_cat[df_cat["Label"] == 0]
        df_anti = df_cat[df_cat["Label"] == 1]

        dates_soli = df_soli["file"].apply(get_folder_and_file).apply(lambda x: dict_dates[x].year)
        dates_anti = df_anti["file"].apply(get_folder_and_file).apply(lambda x: dict_dates[x].year)

        data_soli = dates_soli.value_counts().sort_index()
        data_anti = dates_anti.value_counts().sort_index()

        x_data, y_data = [el for el in range(1867, 2021)], []

        last_val_soli, last_val_anti = 1, 1

        for i in x_data:
            if i in data_soli.index:
                if i in data_anti.index:
                    y_data.append(data_soli.loc[i] / data_anti.loc[i])
                    last_val_anti = data_anti.loc[i]
                else:
                    y_data.append(data_soli.loc[i] / last_val_anti)
                last_val_soli = data_soli.loc[i]
            else:
                if i in data_anti.index:
                    y_data.append(data_anti.loc[i] / last_val_anti)
                    last_val_anti = data_anti.loc[i]
                else:
                    y_data.append(last_val_soli / last_val_anti)

        ax.set_yscale('log')
        ax.plot(x_data, y_data, label=keyword)

    ax.set_xlabel("Year")
    ax.set_ylabel("#Soli / #Anti sentences")

    ax.grid(True, which="both")
    ax.legend()

    plt.axhline(y=1, color='black', linestyle='-')
    plt.suptitle("Relation of solidarity and anti-solidarity sentences expressed in german parliamentary proceedings")
    plt.tight_layout()
    plt.savefig(join(getcwd(), "Frequency_graphs", "auto-annotated-solidarity-relation"))
    plt.show()


def plot_straight():
    df = pd.read_csv(join(PATH, FILE))
    # file: date
    with open(join("dicts", 'dicts/dates.pkl'), 'rb') as _f:
        dict_dates = pickle.load(_f)
    # date: number of sentences
    with open(join("dicts", 'dicts/lengths.pkl'), 'rb') as _f:
        dict_lengths = pickle.load(_f)

    fig, ax = plt.subplots(1, figsize=(10, 4))

    for i, keyword in enumerate(["woman"]):
        df_cat = df[df["category"] == keyword]

        df_soli = df_cat[df_cat["Label"] == 0]
        df_anti = df_cat[df_cat["Label"] == 1]

        #dates_soli = df_soli["file"].apply(get_folder_and_file).apply(lambda x: dict_dates[x])
        dates_anti = df_anti["file"].apply(get_folder_and_file).apply(lambda x: dict_dates[x])

        #data_soli = dates_soli.value_counts().sort_index()
        data_anti = dates_anti.value_counts().sort_index()

        relative_counts_soli, relative_counts_anti = [], []

        #for date, count in zip(data_soli.index, data_soli):
            #relative_counts_soli.append(count/dict_lengths[date.year])
        for date, count in zip(data_anti.index, data_anti):
            relative_counts_anti.append(count/dict_lengths[date.year])

        #ax[i].plot(data_soli.index, relative_counts_soli, label="Solidarity")
        ax.plot(data_anti.index, relative_counts_anti, label="Anti-solidarity")

        ax.set_xlabel("year")
        ax.set_ylabel("# anti-soli / sentences")

        ax.grid(True)
        ax.legend()

    ax.set_title("Women")

    plt.suptitle("Amount of anti-solidarity expressed towards women in german parliamentary proceedings")
    plt.tight_layout()
    plt.savefig(join(getcwd(), "Frequency_graphs", "auto-annotated-solidarity-count-women_anti"))
    plt.show()


def get_folder_and_file(filepath):
    _path, _file = split(filepath)
    _, _folder = split(_path)
    return join(_folder, _file)


if __name__ == "__main__":
    main()
