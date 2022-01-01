import pickle

import matplotlib.pyplot as plt
import pandas as pd

from os import getcwd
from os.path import join
from nltk import word_tokenize

from collections import defaultdict

# 4 year periods (4 year due to later legislative 4 yea rperiods)
x_labels = ["1867", "1871", "1875", "1879", "1883", "1887", "1890", "1894", "1898", "1902", "1906", "1910", "1914",
            "1918", "1923", "1927", "1931", "1932", "1949", "1953", "1957", "1961", "1965", "1969", "1973",
            "1977", "1981", "1982", "1986", "1990", "1994", "1998", "2002", "2005", "2009", "2013", "2017"]

fake_x_values = list(range(len(x_labels)))
BAR_WIDTH = 0.2

keywords = {
    "migrant": {"migration", "vertriebener", "zuwanderung",
                "zuwanderer", "zustrom", "einwanderer", "einwanderung", "ansiedler",
                "aussiedler", "asylbewerber"},
    "frau_nofrau": {"frauen", "maedchen", "mutter", "fraeulein", "genossin", "ehefrau"},
}


def main():
    plot_individuals()
    #plot_singles()
    #plot_together()
    #plot_word_totals()


def plot_individuals():
    """
    for each category of keywords it plots the individual words over time.
    """
    # date: number of sentences
    with open(join("dicts", 'dicts/lengths.pkl'), 'rb') as _f:
        dict_lengths = pickle.load(_f)

    df = pd.read_csv(join(getcwd(), "annotations", "corpus_unannotated_auto-annotated_full.csv"))

    # year: {"word": 4782, "worders":57,...}
    dict_women = defaultdict(defaultdict_with_counter)
    dict_migrants = defaultdict(defaultdict_with_counter)

    # fill data

    df_women = df[df["category"] == "woman"]
    df_migrant = df[df["category"] == "migrant"]

    for sentence, year in zip(df_women["sentence"], df_women["year"]):
        sentence = word_tokenize(sentence)
        for word in sentence:
            if word.lower() in keywords["frau_nofrau"]:
                dict_women[year][word.lower()] += 1
    for sentence, year in zip(df_migrant["sentence"], df_migrant["year"]):
        sentence = word_tokenize(sentence)
        for word in sentence:
            if word.lower() in keywords["migrant"]:
                dict_migrants[year][word.lower()] += 1

    # plot data

    fig, ax = plt.subplots(1, figsize=(10, 5))

    # Women
    bottoms = [0 for _ in dict_women.items()]
    for keyword in keywords["frau_nofrau"]:
        x_vals, y_vals = [], []
        for year, dic in sorted(dict_women.items(), key=lambda x: x[0]):
            x_vals.append(year)
            y_vals.append(dict_women[year][keyword] / dict_lengths[year])

        ax.bar(x_vals, y_vals, bottom=bottoms, label=keyword)
        bottoms = [a+b for a, b in zip(y_vals, bottoms)]

    ax.set_title("Women")
    ax.set_xlabel("Year")
    ax.set_ylabel("Relevant sentences / All sentences")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Number of sentences that relate to a keyword for each year")
    plt.savefig("Frequency_graphs/keyword_occurrences_individuals-bar-women")
    plt.show()

    fig, ax = plt.subplots(1, figsize=(10, 5))

    # Migrants
    bottoms = [0 for _ in dict_migrants.items()]
    for keyword in keywords["migrant"]:
        x_vals, y_vals = [], []
        for year, dic in sorted(dict_migrants.items(), key=lambda x: x[0]):
            x_vals.append(year)
            y_vals.append(dic[keyword] / dict_lengths[year])

        print(f"{keyword=}{sum(y_vals)}")
        ax.bar(x_vals, y_vals, bottom=bottoms, label=keyword)
        bottoms = [a + b for a, b in zip(y_vals, bottoms)]

    ax.set_title("Migrants")
    ax.set_xlabel("Year")
    ax.set_ylabel("Relevant sentences / All sentences")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Number of sentences that relate to a keyword for each year")
    plt.savefig("Frequency_graphs/keyword_occurrences_individuals-bar-migrant")
    plt.show()


def defaultdict_with_counter():
    return defaultdict(int)


def plot_singles():
    """
    Plots each keyword seperately
    2 subplots, 1 for total Frequency, the other for relative Frequency
    """
    # date: number of sentences
    with open(join("dicts", 'dicts/lengths.pkl'), 'rb') as _f:
        dict_lengths = pickle.load(_f)

    # date: number of occurrences
    with open(join("dicts", "dicts/occurrences_women.pkl"), "rb") as f:
        dict_women = pickle.load(f)

    # date: number of occurrences
    with open(join("dicts", "dicts/occurrences_migrants.pkl"), "rb") as f:
        dict_migrants = pickle.load(f)

    fig, ax = plt.subplots(2, figsize=(10, 8))

    # Women
    x_vals, y_vals = [], []
    for year, occs in sorted(dict_women.items(), key=lambda x: x[0]):
        x_vals.append(year)
        y_vals.append(occs/dict_lengths[year])

    ax[0].bar(x_vals, y_vals)
    ax[0].set_title("Women")
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("Relevant sentences / All sentences")
    ax[0].grid(True)
    ax[0].legend()

    # Migrants
    x_vals, y_vals = [], []
    for year, occs in sorted(dict_migrants.items(), key=lambda x: x[0]):
        x_vals.append(year)
        y_vals.append(occs/dict_lengths[year])

    ax[1].bar(x_vals, y_vals)
    ax[1].set_title("Migrants")
    ax[1].set_xlabel("Year")
    ax[1].set_ylabel("Relevant sentences / All sentences")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle("Number of sentences that relate to a keyword for each year")
    plt.savefig("Frequency_graphs/keyword_occurrences-bar")
    plt.show()


def plot_together():
    """ DEPRECATED
    Plots all keywords in 1 graph as a bar graph
    2 subplots, 1 for total Frequencies, other for relative Frequencies
    """
    # load file
    with open(join("dicts", 'dicts/keyword_freq.pkl'), 'rb') as _f:
        dict_keywords = pickle.load(_f)

    fig, ax = plt.subplots(2, figsize=(10, 8))

    for i, _keyword in enumerate(dict_keywords):
        ax[0].bar([el - len(dict_keywords) / 4 * BAR_WIDTH + i * BAR_WIDTH for el in fake_x_values],
                  dict_keywords[_keyword], label=_keyword, width=BAR_WIDTH)

    ax[0].set_xticks(range(len(x_labels)))
    ax[0].set_xticklabels(x_labels)
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("Total occurrences")

    # relative freq bar
    with open(join("dicts", 'dicts/word_totals.pkl'), 'rb') as _f:
        word_totals = pickle.load(_f)

    for i, _keyword in enumerate(dict_keywords):
        ax[1].bar([el - len(dict_keywords) / 4 * BAR_WIDTH + i * BAR_WIDTH for el in fake_x_values],
                  [a / b for a, b in zip(dict_keywords[_keyword], word_totals)], label=_keyword, width=BAR_WIDTH)

    ax[1].set_xticks(range(len(x_labels)))
    ax[1].set_xticklabels(x_labels)
    ax[1].set_xlabel("Year")
    ax[1].set_ylabel("Relative occurrences")
    ax[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.savefig("Frequency_graphs/all_keyword_frequencies")
    plt.show()


def plot_word_totals():
    """
    Plots the amount of words in the corpus for each 4 year period
    """
    # date: number of sentences
    with open(join("dicts", 'dicts/lengths.pkl'), 'rb') as _f:
        lengths = pickle.load(_f)

    fig, ax = plt.subplots(1, figsize=(8, 5))
    x_vals, y_vals = [], []
    for year, occs in sorted(lengths.items(), key=lambda x: x[0]):
        x_vals.append(year)
        y_vals.append(occs)
    ax.bar(x_vals, y_vals)
    ax.set_title("Total number of sentences in the DeuParl corpus each year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of sentences")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("Frequency_graphs/total_sentences")
    plt.show()


if __name__ == "__main__":
    main()

