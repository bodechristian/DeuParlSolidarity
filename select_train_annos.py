from os.path import join

import pandas as pd
import numpy as np

PATH = "annotations"
FILE_NAME_IN = "corpus_unannotated.csv"
FILE_NAME_OUT = "corpus_anno15.csv"
CHUNK_SIZE = 30
TOTAL_DATAPOINTS = 16
PERIODS = ["..\\reichstag_corpora\\kaiserreich_1", "..\\BRD Protokolle\\1\\1_Wahlperiode_TXT_sents",
           "..\\BRD Protokolle\\2\\2_Wahlperiode_TXT_sents", "..\\BRD Protokolle\\3\\3_Wahlperiode_TXT_sents",
           "..\\BRD Protokolle\\4\\4_Wahlperiode_TXT_sents", "..\\BRD Protokolle\\5\\5_Wahlperiode_TXT_sents",
           "..\\BRD Protokolle\\6\\6_Wahlperiode_TXT_sents", "..\\BRD Protokolle\\7\\7_Wahlperiode_TXT_sents",
           "..\\BRD Protokolle\\8\\8_Wahlperiode_TXT_sents", "..\\BRD Protokolle\\9\\9_Wahlperiode_TXT_sents", ]


def main():
    #temporal_cherry_picking()
    default_selection()


def default_selection():
    df = pd.read_csv(join(PATH, FILE_NAME_IN))
    annos_chunked = df.sample(n=CHUNK_SIZE)
    df2 = pd.read_csv(join(PATH, FILE_NAME_OUT))
    df2 = df2.append(annos_chunked)
    df2.to_csv(join(PATH, FILE_NAME_OUT), index=False)


def temporal_cherry_picking():
    df = pd.read_csv(join(PATH, FILE_NAME_IN))
    random_ints_total = []
    for period in PERIODS:
        total_ints = []
        while len(total_ints) < TOTAL_DATAPOINTS:
            random_ints = np.random.choice(len(df), size=CHUNK_SIZE, replace=False)
            annos_chunked = df.loc[random_ints]
            second_indexes = []
            for i, el in enumerate(annos_chunked["file"]):
                if period == "\\".join(el.split("\\")[:-1]):
                    second_indexes.append(i)
            total_ints += [random_ints[i] for i in second_indexes]
            print(f"Datapoints for {period}: {len(total_ints)}")

        random_ints_total.extend(total_ints[:TOTAL_DATAPOINTS])
        print(f"{period} done.")

    annos_chunked = df.loc[random_ints_total]
    annos_chunked.to_csv(join(PATH, FILE_NAME_OUT), index=False)


if __name__ == "__main__":
    main()
