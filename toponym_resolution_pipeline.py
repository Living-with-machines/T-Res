from utils import ner, candidate_selection, linking, eval
from sklearn.model_selection import train_test_split
from transformers import pipeline
import pandas as pd


# Path to NER Model:
ner_model = "/resources/develop/mcollardanuy/toponym-resolution/outputs/models/lwm-ner.model"

# Path to test dataframe:
df = pd.read_csv("/resources/develop/mcollardanuy/toponym-resolution/outputs/data/linking_lwm_df_test.tsv", sep="\t")

# Split test set into dev and test set:
dev, test = train_test_split(df, test_size=0.5, random_state=42)
ner_pipe = pipeline("ner", model=ner_model)
cand_select_method = 'perfect_match' # either perfect_match or deezy_match

dAnnotated, dSentences = ner.format_for_ner(dev)

preds = []
trues = []
for sent_id in dSentences:

    # Toponym recognition
    gold_standard, predictions = ner.ner_predict(dSentences[sent_id], dAnnotated[sent_id], ner_pipe)
    sentence_preds = [[x["word"], x["entity"], "O"] for x in predictions]
    sentence_trues = [[x["word"], x["entity"], x["link"]] for x in gold_standard]

    pred_mentions_sents = ner.aggregate_mentions(sentence_preds)
    true_mentions_sents = ner.aggregate_mentions(sentence_trues)

    # Candidate selection
    mentions = list(set([mention['mention'] for mention in pred_mentions_sents]))
    cands = candidate_selection.select(mentions,cand_select_method)
    
    # # Toponym resolution
    sent = pred_mentions_sents
    for mention in sent:
        text_mention = mention['mention']
        start_offset = mention['start_offset']
        end_offset = mention['end_offset']

        # to be extended so that it can include multiple features and can consider sentence / document context
        res = linking.select(cands[text_mention],'most_popular')
        if res:
            link,score,other_cands = res
            for x in range(start_offset,end_offset+1):
                position_ner = sentence_preds[x][1][:2]
                sentence_preds[x][2] = position_ner+link
                sentence_preds[x].append(other_cands)

    preds.append(sentence_preds)
    trues.append(sentence_trues)


### Assessment of NER

#The SemEval’13 introduced four different ways to measure precision/recall/f1-score results based on the metrics defined by MUC.

#Strict: exact boundary surface string match and entity type;

#Exact: exact boundary match over the surface string, regardless of the type;

#Partial: partial boundary match over the surface string, regardless of the type;

#Type: some overlap between the system tagged entity and the gold annotation is required;

####
print ("\nNER Evaluation")

ner_trues = [[x[1] for x in x] for x in trues]
ner_preds = [[x[1] for x in x] for x in preds]

ner_labels = ['LOC', 'STREET', 'BUILDING']

evaluator = eval.Evaluator(ner_trues, ner_preds, ner_labels)
results = evaluator.evaluate()

for res,scores in results[0].items():
    print (res,"p:",scores["precision"],"r:",scores["recall"])
print (" ")

# print ("LOC")
# for res,scores in results_agg["LOC"].items():
#     print (res,"p:",scores["precision"],"r:",scores["recall"])

# print ("BUILDING")
# for res,scores in results_agg["BUILDING"].items():
#     print (res,"p:",scores["precision"],"r:",scores["recall"])

# print ("STREET")
# for res,scores in results_agg["STREET"].items():
#     print (res,"p:",scores["precision"],"r:",scores["recall"])

# Assessment of candidate selection (this sets the skiline for EL recall)
# cand_sel_score = eval.eval_selection(true_mentions_sents,trues,preds)

# print ('Only in {perc_cand}% of the cases we have retrieved the correct entity among the candidates.\n'.format(perc_cand=cand_sel_score*100))

# Assessment of resolution
print ("EL Evaluation")

all_ids = [y[2] for x in trues for y in x] + [y[2] for x in preds for y in x]
all_ids = list(set([x.split("-")[1] if "-" in x else x for x in all_ids]))
all_ids.remove('O')

el_trues = [[x[2] for x in x] for x in trues]
el_preds = [[x[2] for x in x] for x in preds]

evaluator = eval.Evaluator(el_trues, el_preds, all_ids)
results = evaluator.evaluate()

for res,scores in results[0].items():
    print (res,"p:",scores["precision"],"r:",scores["recall"])