import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import AdamW, BertForSequenceClassification
from tqdm import trange
from os import getcwd
from os.path import join
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PATH = "annotations"
FILE = "corpus_annotated_dataaugmented"

# Parameters
LEARNING_RATE = 1e-5
EPSILON = 1e-8
BATCH_SIZE = 4
EPOCHS = 15

MODEL_NAME = "bert-base-german-dbmdz-uncased"

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name(0)


def main(do_train=False):
    torch.cuda.empty_cache()

    _df = pd.read_csv(join(PATH, FILE+".csv"))

    texts = _df["sentence"].tolist()
    labels = _df["Label"].tolist()

    input_ids, attention_masks = preprocess(texts)

    # split train into train + val
    train_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2021, test_size=0.1)
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels,
                                                                          random_state=2021, test_size=0.1)

    # create dataloaders
    train_dataloader = get_dataloader(train_inputs, train_labels, train_masks, random=True)
    val_dataloader = get_dataloader(val_inputs, val_labels, val_masks)

    print("Finished setting up data.")

    if do_train:
        train(train_dataloader, val_dataloader)


def train(train_dataloader, val_dataloader):
    # model
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.cuda()

    # BERT fine-tuning parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=EPSILON)

    # Store loss and accuracy for plotting
    train_loss_set, val_accuracies = [], []

    # ----------------------
    # TRAINING

    for _ in trange(EPOCHS, desc="Epoch"):
        # Set our model to training mode
        model.train()
        # Tracking variables
        tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
        # Train the data for one epoch
        for batch in train_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # --------------------
        # VALIDATION

        # Put model in evaluation mode
        model.eval()
        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # Evaluate data for one epoch
        for batch in val_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        val_acc = eval_accuracy / nb_eval_steps
        val_accuracies.append(val_acc)
        print(f"Validation Accuracy: {val_acc}")

    # plot training performance
    fig, ax = plt.subplots(2)
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("Batch")
    ax[0].set_ylabel("Loss")
    ax[0].plot(train_loss_set)

    ax[1].set_title("Validation Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].plot(val_accuracies)

    plt.title("model_master")
    plt.tight_layout()

    Path(join(getcwd(), "model_master")).mkdir(parents=True, exist_ok=True)
    plt.savefig(join(getcwd(), "model_master", f"model_master"))
    torch.save(model.state_dict(), join(getcwd(), "model_master", f"model_master.bin"))

    plt.show()


def preprocess(_texts):
    """
    # preprocess (lowercase, add [CLS] ... [SEP], converts to input_ids, adds padding)
    """
    # lowercase
    _texts = [text.lower() for text in _texts]

    # add seperators
    _texts = ["[CLS] " + text + " [SEP]" for text in _texts]

    # tokenize
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(text[:512]) for text in _texts]

    # input_ids
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    print(len(input_ids))

    # pad
    input_ids = pad_sequences(input_ids, maxlen=128, dtype="long", truncating="post", padding="post")

    # attention mask
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # print development of first sentence
    """print(_texts[0])
    print(tokenized_texts[0])
    print(input_ids[0])
    print(attention_masks[0])"""

    return input_ids, attention_masks


def get_dataloader(_inputs, _labels, _masks, random=False):
    """
    takes the inputs, labels and masks
    converts them to tensors and returns the dataloader
    :return: datalooder
    """
    # Convert all of our data into torch tensors, the required datatype for our model
    _inputs = torch.tensor(_inputs)
    _labels = torch.tensor(_labels)
    _masks = torch.tensor(_masks)

    # Make DataLoader
    _data = TensorDataset(_inputs, _masks, _labels)
    if random:
        _dataloader = DataLoader(_data, sampler=RandomSampler(_data), batch_size=BATCH_SIZE)
    else:
        _dataloader = DataLoader(_data, sampler=SequentialSampler(_data), batch_size=BATCH_SIZE)

    return _dataloader


def flat_accuracy(preds, _labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = _labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


if __name__ == "__main__":
    main(do_train=True)
