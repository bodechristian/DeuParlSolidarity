import torch

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertForSequenceClassification
from os import getcwd, listdir
from os.path import join
from train import preprocess
from collections import Counter

import pandas as pd
import numpy as np


PATH_CORPUS = "annotations"
FILE_CORPUS = "corpus_anno15 - corpus_anno15"

# Parameters for name of model
BATCH_SIZE = 4

AGREEMENT_PERCENTAGE = 0.75

TOKENIZER_NAME = "bert-base-german-dbmdz-uncased"
MODELS = [(folder, file_name)
          for folder in listdir(join(getcwd(), "model"))
          for file_name in listdir(join(getcwd(), "model", folder))
          if file_name.endswith(".bin")]

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name(0)


def main():
    torch.cuda.empty_cache()

    df = pd.read_csv(join(PATH_CORPUS, FILE_CORPUS + ".csv"))

    texts = df["sentence"].tolist()

    input_ids, attention_masks = preprocess(texts)

    #all_preds = []
    #for folder, model_name in MODELS:
        #print(f"Predicting with {model_name}")
    torch.cuda.empty_cache()
    # create dataloaders
    # Convert all data into torch tensors, the required datatype for the model
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    # Make DataLoader
    data = TensorDataset(inputs, masks)
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=BATCH_SIZE)

    # model
    model = BertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=3)
    model.load_state_dict(torch.load(join(getcwd(), "model", "lr1e-05-eps1e-08-bs4-epochs25", "lr1e-05-eps1e-08-bs4-epochs25_2.bin")))
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
        # Store predictions and true labels
        predictions.append(logits)
    flat_preds = np.concatenate([np.argmax(pred, axis=1) for pred in predictions]).ravel()
    #all_preds.append(flat_preds)
    print(flat_preds)
    #df["Label"] = flat_preds

    """# take labels where 75%+ agree on it (ensemble)
    resulting_labels = []
    for labels in zip(*all_preds):
        c = Counter(labels)
        label, occurrences = c.most_common(1)[0]
        # for 6 labels and 0.75 agreement %, that means 5 out of 6 have to agree
        if occurrences > AGREEMENT_PERCENTAGE * sum(c.values()):
            resulting_labels.append(label)
        else:
            # use 5 as "undecided" label
            resulting_labels.append(5)"""


    #df["Label"] = resulting_labels

    from sklearn import metrics
    flat_labels = df["Label"]
    # Performance calculations
    precision = metrics.precision_score(flat_labels, flat_preds, average=None)
    recall = metrics.recall_score(flat_labels, flat_preds, average=None)
    f1 = metrics.f1_score(flat_labels, flat_preds, average=None)
    f1_score_micro = metrics.f1_score(flat_labels, flat_preds, average='micro')
    f1_score_macro = metrics.f1_score(flat_labels, flat_preds, average='macro')
    confusion_matrix = metrics.confusion_matrix(flat_labels, flat_preds)

    # Performance prints
    print(f"Accuracy: {metrics.accuracy_score(flat_labels, flat_preds)}")
    print(f'Precision:{precision} \nRecall:{recall} \nF1: {f1}')
    print(f"F1 Score (Micro) = {f1_score_micro} \nF1 Score (Macro) = {f1_score_macro}")
    print(f"Confusion Matrix:\n{confusion_matrix}")
    #df.to_csv(join(PATH_CORPUS, FILE_CORPUS + "_auto-annotated.csv"), index=False)


if __name__ == "__main__":
    main()
