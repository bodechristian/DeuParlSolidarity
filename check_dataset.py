import pandas as pd
import spacy

from collections import Counter
from os import getcwd
from os.path import join
from nltk import word_tokenize

from matplotlib import pyplot as plt

PATH = "annotations"
FILE = "corpus_annotated"
nlp = spacy.load('de_core_news_md')

keywords = {
    "migrant": {"migrant", "migrantinnen", "migration", "immigrant", "fluechtlinge", "vertriebener", "zuwanderung",
                "zuwanderer", "zustrom", "einwanderer", "einwanderung", "auslaender", "auslaenderin", "ansiedler",
                "aussiedler", "asylsuchender", "asylbewerber"},
    "frau_nofrau": {"frauen", "maedchen", "mutter", "fraeulein", "genossin", "ehefrau"}
}


def main():
    df = pd.read_csv(join(getcwd(), PATH, FILE + ".csv"))

    #df = erase3s(df)
    #draw_temporal_distribution(df)
    check_class_distribution(df)
    check_mig_frau_distribution(df)
    #save_txts(df)


def erase3s(_df):
    """looks for 3s in the PATH, FILE and replaces them with 2s"""
    _df["Label"] = _df["Label"].replace(3, 2)
    _df.to_csv(join(getcwd(), PATH, FILE + ".csv"))
    return _df


def draw_temporal_distribution(_df):
    b = Counter([el for el in _df["file"]])
    data = Counter()
    for el, val in b.items():
        data[el.split("\\")[-2]] += val
    data = data.items()
    data = sorted(data, key=lambda x: x[0])
    data = sorted(data, key=lambda x: len(x[0]))
    temp = data.pop(0)
    data.insert(2, temp)
    print(data)

    fig, ax = plt.subplots(1, figsize=(8, 5))
    x_data, y_data = [el[0] for el in data], [el[1] for el in data]
    x_data = [el.replace("_TXT_sents", "") for el in x_data]
    ax.bar(x_data, y_data)
    fig.autofmt_xdate()
    plt.xlabel("Time period")
    plt.ylabel("Number of data")
    plt.title("Number of sentences taken from each period for annotation")

    plt.tight_layout()
    plt.savefig(join(getcwd(), "Frequency_graphs", "temporal_distribution"))
    plt.show()


def check_class_distribution(_df):
    a = Counter([int(el) for el in _df["Label"]])
    print(a)
    print(len(_df))


def check_mig_frau_distribution(_df):
    c = Counter()

    for line, label in zip(_df["sentence"], _df["Label"]):
        for word in word_tokenize(line):
            for keyword in keywords:
                if word.lower() in keywords[keyword]:
                    c[keyword] += 1
    print(c)


def save_txts(_df):
    # Solidarity
    with open("word_log_frequency/solidarity-migrant.txt", "w", encoding="utf-8", newline="") as f:
        lst = preprocess(_df[_df["Label"] == 0]["sentence"])
        f.writelines(lst)
        
    # Anti-solidarity
    with open("word_log_frequency/anti-solidarity-migrant.txt", "w", encoding="utf-8", newline="") as f:
        lst = preprocess(_df[_df["Label"] == 1]["sentence"])
        f.writelines(lst)


def preprocess(iterable):
    #names = ["frau_nofrau", "migrant"]
    lst = []
    for line in iterable:
        add = False
        for word in word_tokenize(line):
            if word.lower() in keywords["migrant"]:
                add = True
        if add:
            lst.append(" ".join([word.lemma_ for word in nlp(line)]))
    return lst



if __name__ == "__main__":
    main()
