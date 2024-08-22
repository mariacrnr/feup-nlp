import pandas as pd

de_train = pd.read_csv("./train_dev_test_splits/de.train.csv", sep="\t")
de_val = pd.read_csv("./train_dev_test_splits/de.valid.csv", sep="\t")
de_test = pd.read_csv("./train_dev_test_splits/de.test.csv", sep="\t")

fr_train = pd.read_csv("./train_dev_test_splits/fr.train.csv", sep="\t")
fr_val = pd.read_csv("./train_dev_test_splits/fr.valid.csv", sep="\t")
fr_test = pd.read_csv("./train_dev_test_splits/fr.test.csv", sep="\t")

drop_cols = list(fr_train.columns)
drop_cols.remove("content")
drop_cols.remove("e1")

fr_train.drop(columns=drop_cols, inplace=True)
fr_val.drop(columns=drop_cols, inplace=True)
fr_test.drop(columns=drop_cols, inplace=True)

fr_train.rename(columns={'e1':'label'}, inplace = True)
fr_val.rename(columns={'e1':'label'}, inplace = True)
fr_test.rename(columns={'e1':'label'}, inplace = True)

from datasets import Dataset

de_train = Dataset.from_pandas(de_train)
de_val = Dataset.from_pandas(de_val)
de_test = Dataset.from_pandas(de_test)

fr_train = Dataset.from_pandas(fr_train)
fr_val = Dataset.from_pandas(fr_val)
fr_test = Dataset.from_pandas(fr_test)

from datasets import DatasetDict

# gather everyone if you want to have a single DatasetDict
train_valid_test_dataset_de = DatasetDict({
    'train': de_train,
    'validation': de_val,
    'test': de_test
})

train_valid_test_dataset_fr = DatasetDict({
    'train': fr_train,
    'validation': fr_val,
    'test': fr_test
})

model_name_de = "distilbert-base-multilingual-cased"
model_name_fr = "distilbert-base-multilingual-cased"
#model_name_fr = "camembert-base"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name_fr)

def preprocess_function(sample):
    return tokenizer(sample["content"], truncation=True)

tokenized_dataset = train_valid_test_dataset_fr.map(preprocess_function, batched=True)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name_fr, num_labels=3)

from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_metric
import numpy as np

metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model()
trainer.predict(test_dataset=tokenized_dataset["test"])

import torch

y_pred= []
for p in tokenized_dataset['test']['content']:
    ti = tokenizer(p, return_tensors="pt")
    out = model(**ti)
    pred = torch.argmax(out.logits)
    y_pred.append(pred)   # our labels are already 0 and 1

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

y_test = tokenized_dataset['test']['label']

print(confusion_matrix(y_test, y_pred))
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred, average='macro'))
print('Recall: ', recall_score(y_test, y_pred, average='macro'))
print('F1: ', f1_score(y_test, y_pred, average='macro'))