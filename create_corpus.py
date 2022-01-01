import pandas as pd

from os.path import join
from os import listdir
from nltk import word_tokenize
from collections import defaultdict

PATH = ".."
FILE_PATH_OUT = "annotations"
FILE_NAME_OUT = "corpus_unannotated.csv"
WINDOW = 4

REIGN_ORDER_RT = ["kaiserreich_1", "kaiserreich_2", "weimar", "ns"]
keywords = {
    "migrant": {"migrant", "migrantinnen", "migration", "immigrant", "fluechtlinge", "vertriebener", "zuwanderung",
                "zuwanderer", "zustrom", "einwanderer", "einwanderung", "auslaender", "auslaenderin", "ansiedler",
                "aussiedler", "asylsuchender", "asylbewerber"},
    "frau_nofrau": {"frauen", "maedchen", "mutter", "fraeulein", "genossin", "ehefrau"},
}


def main():
    rows = defaultdict(list)

    # ../reichstag_corpora/[REIGN_ORDER_RT]/1_sents.txt
    rt_path = join(PATH, "reichstag_corpora")
    for reign_period in listdir(rt_path):
        print(f"starting {reign_period}")
        reign_path = join(rt_path, reign_period)
        for file in listdir(reign_path):
            if file == ".DS_Store":
                continue
            file_path = join(reign_path, file)
            with open(file_path, "r", encoding='utf-8') as f:
                text = f.readlines()
                for i, line in enumerate(text):
                    line_added = False
                    for word in word_tokenize(line):
                        if line_added:
                            continue
                        for keyword in keywords:
                            if word.lower() in keywords[keyword]:
                                _dic = {
                                    "file": file_path,
                                    "prev_sents": " ".join(text[i - WINDOW:i]),
                                    "sentence": line,
                                    "next_sents": " ".join(text[i + 1: i + WINDOW + 1])
                                }
                                rows[keyword].append(_dic)
                                line_added = True
                                continue
        print(f"finished {reign_period}")

    # ../BRD Protokolle/[1..19]/[1..19]_Wahlperiode_TXT_sents/[.*].txt
    brd_path = join(PATH, "BRD Protokolle")
    for leg_period in listdir(brd_path):
        print(f"starting {leg_period}")
        leg_path = join(brd_path, leg_period, leg_period + "_Wahlperiode_TXT_sents")
        for file in listdir(leg_path):
            if file == ".DS_Store":
                continue
            file_path = join(leg_path, file)
            with open(file_path, "r", encoding='utf-8') as f:
                text = f.readlines()
                for i, line in enumerate(text):
                    line_added = False
                    for word in word_tokenize(line):
                        if line_added:
                            continue
                        for keyword in keywords:
                            if word.lower() in keywords[keyword]:
                                _dic = {
                                    "file": file_path,
                                    "prev_sents": " ".join(text[i - WINDOW:i]),
                                    "sentence": line,
                                    "next_sents": " ".join(text[i + 1: i + WINDOW + 1])
                                }
                                rows[keyword].append(_dic)
                                line_added = True
                                continue
        print(f"finished {leg_period}")
    for keyword in keywords:
        pd.DataFrame(rows[keyword]).to_csv(join(FILE_PATH_OUT, "corpus_unannotated_"+keyword+".csv"))


if __name__ == "__main__":
    main()
