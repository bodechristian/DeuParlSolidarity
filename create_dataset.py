from os import getcwd, listdir
from os.path import join
import pandas as pd

PATH = join(getcwd(), "annotations", "400")


def main():
    all_dfs = []
    for filename in listdir(PATH)[:10]:
        df_temp = pd.read_csv(join(PATH, filename))
        all_dfs.append(df_temp[["id", "file", "prev_sents", "sentence", "next_sents", "Label"]])
    df = pd.concat([_df for _df in all_dfs], ignore_index=True)
    print(len(df))
    df.to_csv(join(getcwd(), "annotations", "corpus_annotated-400.csv"), index=False)


if __name__ == "__main__":
    main()
