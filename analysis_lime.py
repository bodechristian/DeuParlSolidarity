import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from os.path import join
from os import getcwd
from lime.lime_text import LimeTextExplainer
from train import preprocess

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name(0)

TOKENIZER_NAME = "bert-base-german-dbmdz-uncased"


def main():
    torch.cuda.empty_cache()

    df = pd.read_csv(join("annotations", "annotated", "corpus_anno15 - corpus_anno15.csv"))
    texts = df["sentence"].tolist()
    labels = df["Label"].tolist()
    idxs = [3]

    word = "ich will keine migranten hier im land"
    sol = predict_proba([word])

    print(f"predicting: < {word} >")
    print("prediction:", sol)
    print(f"correct label: {np.argmax(sol)} - {['Solidary', 'Anti-Solidary', 'Other'][np.argmax(sol)]}")
    """explainer = LimeTextExplainer(class_names=["Solidarity", "Anti-Solidarity", "Other"])

    for idx in idxs:
        exp = explainer.explain_instance(texts[idx], predict_proba, num_features=6, labels=[0, 1])
        print("prediction:", predict_proba([texts[idx]]))
        print("label:", labels[idx])
        print("text", texts[idx])
        exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(join(getcwd(), "lime", f"lime-{idx}"))
        exp.save_to_file(join(getcwd(), "lime", f"lime-{idx}-temp.html"))"""


def predict_proba(datapoints):
    ids, masks = preprocess(datapoints)

    input_ids = torch.tensor(ids)
    masks = torch.tensor(masks)

    # Make DataLoader
    data = TensorDataset(input_ids, masks)
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=4)

    # model
    model = BertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=3)
    model.load_state_dict(torch.load(join(getcwd(), "model_master", "model_master.bin")))
    model.cuda()
    model.eval()

    # Predict
    predictions = np.empty((0, 3))
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
        predictions = np.append(predictions, logits_norm, axis=0)
    return predictions


if __name__ == "__main__":
    main()
