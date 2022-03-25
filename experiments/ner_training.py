from datasets import load_metric
from datasets import load_dataset
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
from argparse import ArgumentParser

# Code adapted from HuggingFace tutorial: https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb


# -------------------
# Input and output parameters:
output_ner_model = "outputs/models/" # Path where NER model will be stored.
model_name = "lwm-ner" # NER model name.
base_model_path = "/resources/models/bert/bert_1760_1900/" # Path where base models are stored (it can be
                                                           # a pretrained Huggingface model as well, such
                                                           # as 'distilbert-base-uncased'. The 'bert_1760_1900'
                                                           # model can be obtained following instructions in
                                                           # https://github.com/Living-with-machines/histLM).
ner_data_path = "outputs/data/lwm/" # Path where preprocessed NER data is stored.
train_dataset = 'ner_df_train.json' # NER training set (0.80 of LwM training set).
test_dataset = 'ner_df_dev.json' # NER test set (0.20 of LwM training set).


Path(output_ner_model).mkdir(parents=True, exist_ok=True)
task = "ner"
metric = load_metric("seqeval")


# -------------------
# Test option
parser = ArgumentParser()

parser.add_argument('-t',
                    dest="test",
                    choices=('True','False'),
                    help='Run in test mode. Options are "True" or "False", default is "True".',
                    default='True')

args = parser.parse_args()


# -------------------
# Fine-tuning for NER

if args.test == 'True':
    # If test is True, train on 5% of the train and test sets, and add "_test" to the model name.
    model_name = model_name + "_test"
    lwm_train = load_dataset('json', data_files=ner_data_path + train_dataset, split='train[:5%]')
    lwm_test = load_dataset('json', data_files=ner_data_path + test_dataset, split='train[:5%]')
else:
    lwm_train = load_dataset('json', data_files=ner_data_path + train_dataset, split='train')
    lwm_test = load_dataset('json', data_files=ner_data_path + test_dataset, split='train')
    

label_encoding_dict = {'O': 0,
                       'B-LOC': 1,
                       'I-LOC': 2,
                       'B-STREET': 3,
                       'I-STREET': 4,
                       'B-BUILDING': 5,
                       'I-BUILDING': 6,
                       'B-OTHER': 7,
                       'I-OTHER': 8,
                       'B-FICTION': 9,
                       'I-FICTION': 10}
label_list = list(label_encoding_dict.keys())


# Align tokens and labels:
def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Compute metrics:
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


model = AutoModelForTokenClassification.from_pretrained(base_model_path, num_labels=len(label_list))
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
data_collator = DataCollatorForTokenClassification(tokenizer)

lwm_train_tok = lwm_train.map(tokenize_and_align_labels, batched=True)
lwm_test_tok = lwm_test.map(tokenize_and_align_labels, batched=True)

training_args = TrainingArguments(
    output_dir=output_ner_model,
    evaluation_strategy="epoch",
    logging_dir=output_ner_model + "runs/" + model_name,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lwm_train_tok,
    eval_dataset=lwm_test_tok,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()

trainer.evaluate()

trainer.save_model(output_ner_model + model_name + '.model')