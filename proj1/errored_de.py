import json
from googletrans import Translator
import pandas as pd

file = open(f'./lr_tfidf_unigram_de_ov.json')
config = json.load(file)

error = config["results"]["train_results"][0]["misclassified_cases"]

de_train = pd.read_csv("./train_dev_test_splits/de.train.csv", sep="\t")
de_val = pd.read_csv("./train_dev_test_splits/de.valid.csv", sep="\t")
de_test = pd.read_csv("./train_dev_test_splits/de.test.csv", sep="\t")

de = pd.concat([de_train, de_val, de_test])

translator = Translator()
eng_list = []
count = 0
for case in error:
    translation = translator.translate(de.loc[de['id'] == case["case"]]['content'].values[0], dest='en')
    if (case["predicted"] == 0):
        count += 1
    eng_list.append({"text" : translation.text, "predicted" : case["predicted"], "expected" : case["expected"]})


print(count/len(eng_list))
with open(f"./dump.json", 'w') as outfile:
    json.dump({"error" : eng_list}, outfile, indent=2)