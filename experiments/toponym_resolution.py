import sys,os
# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_data, ner, candidate_selection, linking, eval
from sklearn.model_selection import train_test_split
from transformers import pipeline
import pandas as pd
import glob
import tqdm
import json

# Dataset:
dataset = "lwm"

# Approach:
ner_model_id = 'lwm'
cand_select_method = 'perfectmatch' # either perfectmatch or deezymatch
top_res_method = 'mostpopular'

# Path to NER Model:
ner_model = "/resources/develop/mcollardanuy/toponym-resolution/outputs/models/"+ner_model_id+"-ner.model"

# Path to test dataframe:
df = pd.read_csv("/resources/develop/mcollardanuy/toponym-resolution/outputs/data/linking_" + dataset + "_df_test.tsv", sep="\t")

# Path to processed data:
processed_path = "outputs/data/"

# Load all sentences in test:
dSentences = dict()
for fid in glob.glob(processed_path + "test_" + dataset + "_sentences/*"):
    file_id = fid.split("/")[-1].split("_")[0] # Document id
    with open(fid) as json_file:
        data = json.load(json_file)
        dSentences[file_id] = data

# Split test set into dev and test set (by article, not sentence):
dev_ids, test_ids = train_test_split(list(dSentences.keys()), test_size=0.5, random_state=42)
dev = df[df["article_id"].astype(str).isin(dev_ids)]
test = df[df["article_id"].astype(str).isin(test_ids)]

ner_pipe = pipeline("ner", model=ner_model)

dAnnotated, dSentencesDev = ner.format_for_ner(dev, dSentences, dev_ids)

true_mentions_sents = dict()
dPreds = dict()
dTrues = dict()

for sent_id in tqdm.tqdm(dSentencesDev.keys()):

    # Toponym recognition
    gold_standard, predictions = ner.ner_predict(dSentencesDev[sent_id], dAnnotated[sent_id], ner_pipe)

    sentence_preds = [[x["word"], x["entity"], "O"] for x in predictions]
    sentence_trues = [[x["word"], x["entity"], x["link"]] for x in gold_standard]

    pred_mentions_sent = ner.aggregate_mentions(sentence_preds)
    true_mentions_sent = ner.aggregate_mentions(sentence_trues)

    # Candidate selection
    mentions = list(set([mention['mention'] for mention in pred_mentions_sent]))
    cands = candidate_selection.select(mentions,cand_select_method)
    
    # # Toponym resolution
    for mention in pred_mentions_sent:
        text_mention = mention['mention']
        start_offset = mention['start_offset']
        end_offset = mention['end_offset']

        # to be extended so that it can include multiple features and can consider sentence / document context
        res = linking.select(cands[text_mention],top_res_method)
        if res:
            link,score,other_cands = res
            for x in range(start_offset,end_offset+1):
                position_ner = sentence_preds[x][1][:2]
                sentence_preds[x][2] = position_ner+link
                sentence_preds[x].append(other_cands)

    dPreds[sent_id] = sentence_preds
    dTrues[sent_id] = sentence_trues
    true_mentions_sents[sent_id] = true_mentions_sent

process_data.store_results_hipe(dataset,ner_model_id+'+'+cand_select_method+'+'+top_res_method,  dPreds)
process_data.store_results_hipe(dataset,'true', dTrues)
skyline = eval.eval_selection(true_mentions_sents,dTrues,dPreds)
process_data.store_resolution_skyline(dataset,ner_model_id+'+'+cand_select_method+'+'+top_res_method,skyline)
