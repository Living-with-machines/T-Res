import json
import sys,os
import urllib
# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
# to read the gold standard for rel
#sys.path.insert(0,os.path.abspath(os.path.pardir+'/evaluation/CLEF-HIPE-2020-scorer/'))
#import ner_evaluation.utils

from utils import process_data, ner, candidate_selection, linking, eval
from sklearn.model_selection import train_test_split
from transformers import pipeline
import pandas as pd
import tqdm

# Dataset:
dataset = "lwm"


# Approach:
ner_model_id = 'rel' # or rel

if ner_model_id == 'lwm':
    # Path to NER Model:
    ner_model = "/resources/develop/mcollardanuy/toponym-resolution/outputs/models/"+ner_model_id+"-ner.model"
    ner_pipe = pipeline("ner", model=ner_model)
    cand_select_method = 'perfectmatch' # either perfectmatch or deezymatch
    top_res_method = 'mostpopular'

if ner_model_id == 'rel':
    gold_path = "outputs/results/" + dataset + "/true_bundle2_en_1.tsv"
    gold_standard = process_data.read_gold_standard(gold_path)
    cand_select_method = 'rel' # either perfectmatch or deezymatch
    top_res_method = 'rel'



# Path to test dataframe:
df = pd.read_csv("/resources/develop/mcollardanuy/toponym-resolution/outputs/data/linking_lwm_df_test.tsv", sep="\t")

# Split test set into dev and test set (by article, not sentence):
dev_ids, test_ids = train_test_split(df.article_id.unique(), test_size=0.5, random_state=42)
dev = df[df["article_id"].isin(dev_ids)]
test = df[df["article_id"].isin(test_ids)]


dAnnotated, dSentences = ner.format_for_ner(dev)

true_mentions_sents = dict()
dPreds = dict()
dTrues = dict()

path = '/resources/wikipedia/extractedResources/'
with open(path+'wikipedia2wikidata.json') as f:
    wikipedia2wikidata = json.load(f)

# check NER labels in REL
accepted_labels = {'LOC','STREET','BUILDING','OTHER','FICTION'}

def match_ent(pred_ents,start,end,prev_ann):
    for ent in pred_ents:
        if ent[-1] in accepted_labels:
            st_ent = ent[0]
            if st_ent>= start and st_ent<=end:
                if prev_ann == ent[-1]:
                    ent_pos = 'I-'
                else:
                    ent_pos = 'B-'
                    prev_ann = ent[-1]

                n = ent_pos+ent[-1]
                el =  urllib.parse.quote(ent[3].replace("_"," "))
                try:
                    el = ent_pos+wikipedia2wikidata[el]
                except Exception:
                    # to be checked but it seems some Wikipedia pages are not in Wikidata
                    # see for instance Zante%2C%20California
                    return n, 'O',''
                    #print (el)
                return n,el,prev_ann
    return 'O','O',''

for sent_id in tqdm.tqdm(dSentences.keys()):

    if ner_model_id == 'rel':
        try:
            pred_ents = linking.rel_end_to_end(dSentences[sent_id])

        except Exception as e:
            print (e)
        char_count = 0
        sentence_preds = []
        for token in gold_standard[sent_id]:
            start = char_count
            end = char_count+(len(token)-1)
            prev_ann = ''
            n,el,prev_ann = match_ent(pred_ents,start,end,prev_ann)
            sentence_preds.append([token,n,el])
            char_count = char_count+len(token)
        dPreds[sent_id] = sentence_preds

    else:
        # Toponym recognition
        gold_standard, predictions = ner.ner_predict(dSentences[sent_id], dAnnotated[sent_id], ner_pipe)
        sentence_preds = [[x["word"], x["entity"], "O"] for x in predictions]
        sentence_trues = [[x["word"], x["entity"], x["link"]] for x in gold_standard]

        pred_mentions_sent = ner.aggregate_mentions(sentence_preds)
        true_mentions_sent = ner.aggregate_mentions(sentence_trues)
        print (sentence_preds)
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

if ner_model_id == 'lwm':
    process_data.store_results_hipe(dataset,'true', dTrues)
    skyline = eval.eval_selection(true_mentions_sents,dTrues,dPreds)
    process_data.store_resolution_skyline(dataset,ner_model_id+'+'+cand_select_method+'+'+top_res_method,skyline)

process_data.store_results_hipe(dataset,ner_model_id+'+'+cand_select_method+'+'+top_res_method,  dPreds)
