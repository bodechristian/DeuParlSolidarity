import pickle
import torch

from nltk import word_tokenize
from os.path import join, split
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertForSequenceClassification
from os import getcwd
from sklearn.metrics import accuracy_score
from sklearn import metrics
from train import preprocess
from os import listdir

import pandas as pd
import numpy as np

"""
random extra methods I had to occasionally use
"""

PATH_CORPUS = "annotations"
FILE_CORPUS = "corpus_annotated"

TOKENIZER_NAME = "bert-base-german-dbmdz-uncased"

keywords = {
    "migrant": {"migrant", "migrantinnen", "migration", "immigrant", "fluechtlinge", "vertriebener", "zuwanderung",
                "zuwanderer", "zustrom", "einwanderer", "einwanderung", "auslaender", "auslaenderin", "ansiedler",
                "aussiedler", "asylsuchender", "asylbewerber"},
    "woman": {"frauen", "maedchen", "mutter", "fraeulein", "genossin", "ehefrau"},
}


def main():
    df = pd.read_csv(join(getcwd(), "annotations", "corpus_unannotated_auto-annotated_full.csv"))
    for el in df[df["sentence"].str.contains("flüchtlinge")]["sentence"]:
        print(el)
    #file_to_time(r"..\BRD Protokolle\13\13_Wahlperiode_TXT_sents\210_sents.txt")
    #test_ensemble()
    #get_full_csv()
    #check_class_distribution(13)
    #check_class_distribution(14)
    #get_rows_with_label(1, 30)
    #extract_year(2018, "migrant")


def extract_year(year, cat):
    df = pd.read_csv(join(getcwd(), "annotations", "corpus_unannotated_auto-annotated_full.csv"))
    df = df[df["category"] == cat]
    df = df[df["year"] == year]
    print(len(df))
    df.to_csv(join(getcwd(), "temp.csv"), index=False)


def file_to_time(filename):
    # file: date
    with open(join("dicts", 'dicts/dates.pkl'), 'rb') as _f:
        dict_dates = pickle.load(_f)
    print(dict_dates[get_folder_and_file(filename)])


def test_ensemble():
    AGREEMENT_PERCENTAGE = 0.75
    MODELS = [(folder, file_name)
              for folder in listdir(join(getcwd(), "model"))
              for file_name in listdir(join(getcwd(), "model", folder))
              if file_name.endswith(".bin")]

    # specify GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.get_device_name(0)
    torch.cuda.empty_cache()

    df = pd.read_csv(join(PATH_CORPUS, "annotated", "corpus_anno15 - corpus_anno15.csv"))
    texts = df["sentence"].tolist()
    input_ids, attention_masks = preprocess(texts)

    all_preds = []
    for folder, model_name in MODELS:
        print(f"Predicting with {model_name}")
        torch.cuda.empty_cache()
        # create dataloaders
        # Convert all data into torch tensors, the required datatype for the model
        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)

        # Make DataLoader
        data = TensorDataset(inputs, masks)
        dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=4)

        # model
        model = BertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=3)
        model.load_state_dict(torch.load(join(getcwd(), "model", folder, model_name)))
        model.cuda()
        model.eval()

        # Predict
        predictions = []
        for batch in dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            logits_norm = (np.exp(logits).T / np.exp(logits).sum(-1)).T
            # Store predictions and true labels
            predictions.append(logits_norm)
        #flat_preds = np.concatenate([np.argmax(pred, axis=1) for pred in predictions]).ravel()
        predictions = np.asarray(predictions)
        flat_preds = np.reshape(predictions, (predictions.shape[0]*predictions.shape[1],  predictions.shape[2]))
        all_preds.append(flat_preds)

        preds_ensembled ,labels_ensembled = flat_preds.argmax(axis=1),  df["Label"].tolist()
        # Performance calculations
        precision = metrics.precision_score(labels_ensembled, preds_ensembled, average=None)
        recall = metrics.recall_score(labels_ensembled, preds_ensembled, average=None)
        f1 = metrics.f1_score(labels_ensembled, preds_ensembled, average=None)
        f1_score_micro = metrics.f1_score(labels_ensembled, preds_ensembled, average='micro')
        f1_score_macro = metrics.f1_score(labels_ensembled, preds_ensembled, average='macro')
        confusion_matrix = metrics.confusion_matrix(labels_ensembled, preds_ensembled)

        # Performance prints
        print(f"\nModel:\tEnsemble")
        print(f"Accuracy: {accuracy_score(labels_ensembled, preds_ensembled)}")
        print(f'Precision:{precision} \nRecall:{recall} \nF1: {f1}')
        print(f"F1 Score (Micro) = {f1_score_micro} \nF1 Score (Macro) = {f1_score_macro}")
        print(f"Confusion Matrix:\n{confusion_matrix}")

    # take labels where 75%+ agree on it
    ensembled_preds = []
    for labels in zip(*all_preds):
        temp = np.asarray(labels)
        print("pre-average")
        print(temp)
        temp = np.average(temp, axis=0)
        print("post-average")
        print(temp)
        print(f"max: {temp.argmax()}")
        ensembled_preds.append(temp.argmax())
        """c = Counter(labels)
        label, occurrences = c.most_common(1)[0]
        # for 6 labels and 0.75 agreement %, that means 5 out of 6 have to agree
        if occurrences > AGREEMENT_PERCENTAGE * sum(c.values()):
            ensembled_preds.append(label)
        else:
            # write 5 when undecided
            ensembled_preds.append(5)"""

    df["ensemble-proba"] = ensembled_preds
    #df.to_csv(join(PATH_CORPUS, "annotated", "corpus_anno15 - corpus_anno15.csv"))

    # take out the unsure entries
    preds_ensembled, labels_ensembled = ensembled_preds, df["Label"].tolist()
    """for pred, label in zip(ensembled_preds, df["Label"].tolist()):
        if pred != 5:
            preds_ensembled.append(pred)
            labels_ensembled.append(label)"""

    # Performance calculations
    precision = metrics.precision_score(labels_ensembled, preds_ensembled, average=None)
    recall = metrics.recall_score(labels_ensembled, preds_ensembled, average=None)
    f1 = metrics.f1_score(labels_ensembled, preds_ensembled, average=None)
    f1_score_micro = metrics.f1_score(labels_ensembled, preds_ensembled, average='micro')
    f1_score_macro = metrics.f1_score(labels_ensembled, preds_ensembled, average='macro')
    confusion_matrix = metrics.confusion_matrix(labels_ensembled, preds_ensembled)

    # Performance prints
    print(f"\nModel:\tEnsemble")
    print(f"Accuracy: {accuracy_score(labels_ensembled, preds_ensembled)}")
    print(f'Precision:{precision} \nRecall:{recall} \nF1: {f1}')
    print(f"F1 Score (Micro) = {f1_score_micro} \nF1 Score (Macro) = {f1_score_macro}")
    print(f"Confusion Matrix:\n{confusion_matrix}")


def get_full_csv():
    with open(join("dicts", 'dicts/dates.pkl'), 'rb') as _f:
        dict_dates = pickle.load(_f)

    df = pd.read_csv(join(PATH_CORPUS, "corpus_unannotated_auto-annotated.csv"))
    df["year"] = df["file"].apply(get_folder_and_file).apply(lambda x: dict_dates[x].year)
    df["category"] = df["sentence"].apply(get_category)

    df.to_csv(join(PATH_CORPUS, "corpus_unannotated_auto-annotated_full.csv"), index=False)


def get_category(sent):
    for word in word_tokenize(sent):
        for keyword in keywords:
            if word.lower() in keywords[keyword]:
                return keyword


def get_folder_and_file(filepath):
    _path, _file = split(filepath)
    _, _folder = split(_path)
    return join(_folder, _file)


def check_class_distribution(nb):
    df = pd.read_csv(join(PATH_CORPUS, f"corpus_anno{nb} - corpus_anno{nb}.csv"))
    a = Counter([int(el) for el in df["Label"]])
    print(a)
    print()


def get_rows_with_label(lbl, nb_rows=100):
    _df = pd.read_csv(join(PATH_CORPUS, "corpus_unannotated_auto-annotated-old.csv"))
    _df = _df[_df["Label"] == lbl].sample(n=nb_rows)
    _df.to_csv(join(PATH_CORPUS, "corpus_anno15.csv"), index=False)


if __name__ == "__main__":
    main()
