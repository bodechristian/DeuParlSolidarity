import pandas as pd
import numpy as np

from os import getcwd
from os.path import join


PATH = "annotations"
FILE = "corpus_annotated"


def main():
    df = pd.read_csv(join(getcwd(), PATH, FILE + ".csv"))

    label0 = len(df[df["Label"] == 0])
    label1 = len(df[df["Label"] == 1])
    label2 = len(df[df["Label"] == 2])

    _max = max(label0, label1, label2)

    diff0 = _max - label0
    diff1 = _max - label1
    diff2 = _max - label2

    random_ints0 = np.random.choice(label0, size=diff0)
    random_ints1 = np.random.choice(label1, size=diff1)
    random_ints2 = np.random.choice(label2, size=diff2)

    added0 = df[df["Label"] == 0].iloc[random_ints0]
    added1 = df[df["Label"] == 1].iloc[random_ints1]
    added2 = df[df["Label"] == 2].iloc[random_ints2]

    df_out = pd.concat([df, added0, added1, added2])

    df_out.to_csv(join(PATH, FILE+"_dataaugmented.csv"), index=False)


if __name__ == "__main__":
    main()
