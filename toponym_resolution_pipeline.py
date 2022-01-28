from utils import ner, candidate_selection, linking, eval

# Path to NER Model:
ner_model = "/resources/develop/mcollardanuy/toponym-resolution/outputs/models/lwm-ner.model"

# use NER and obtain the following output (to be added)

preds = [[['mr', 'O', 'O'], ['.', 'O', 'O'], ['oldham', 'O', 'O'], [',', 'O', 'O'], ['ol', 'O', 'O'], ['London', 'B-LOC', 'O'], [',', 'O', 'O'], ['in', 'O', 'O'], ['stafford', 'B-LOC', 'O'], ['(', 'I-LOC', 'O'], ['hire', 'I-LOC', 'O'], [',', 'O', 'O'], ['to', 'O', 'O'], ['mils', 'O', 'O'], ['oldlcaa', 'O', 'O'], [',', 'O', 'O'], ['lie', 'O', 'O'], ['of', 'O', 'O'], ['this', 'O', 'O'], ['town', 'O', 'O'], ['.', 'O', 'O']], [['glo', 'O', 'O'], ['uc', 'O', 'O'], ['est', 'O', 'O'], ['ers', 'O', 'O'], ['h1', 'O', 'O'], ['re', 'O', 'O'], ['.', 'O', 'O']], [['golden', 'B-BUILDING', 'O'], ['lion', 'I-BUILDING', 'O'], ['inn', 'O', 'O'], [',', 'O', 'O'], ['wei', 'B-LOC', 'O'], ['mouth', 'I-LOC', 'O'], ['.', 'O', 'O']], [['mr', 'O', 'O'], ['armit', 'O', 'O'], ['«', 'O', 'O'], ['t', 'O', 'O'], ['.', 'O', 'O'], ['ad', 'O', 'O'], [',', 'O', 'O'], ['of', 'O', 'O'], ['gre', 'B-LOC', 'O'], ['>', 'I-LOC', 'O'], ['s', 'I-LOC', 'O'], [';', 'I-LOC', 'O'], ['nglia', 'I-LOC', 'O'], [',', 'O', 'O'], ['in', 'O', 'O'], ['this', 'O', 'O'], ['county', 'O', 'O'], ['.', 'O', 'O']]]
trues = [[['mr', 'O', 'O'], ['.', 'O', 'O'], ['oldham', 'O', 'O'], [',', 'O', 'O'], ['ol', 'O', 'O'], ['London', 'B-LOC', 'B-Q84'], [',', 'O', 'O'], ['in', 'O', 'O'], ['stafford', 'B-LOC', 'O'], ['(', 'I-LOC', 'O'], ['hire', 'I-LOC', 'O'], [',', 'O', 'O'], ['to', 'O', 'O'], ['mils', 'O', 'O'], ['oldlcaa', 'O', 'O'], [',', 'O', 'O'], ['lie', 'O', 'O'], ['of', 'O', 'O'], ['this', 'O', 'O'], ['town', 'O', 'O'], ['.', 'O', 'O']], [['glo', 'B-LOC', 'B-Q23165'], ['uc', 'I-LOC', 'I-Q23165'], ['est', 'I-LOC', 'I-Q23165'], ['ers', 'I-LOC', 'I-Q23165'], ['h1', 'I-LOC', 'I-Q23165'], ['re', 'I-LOC', 'I-Q23165'], ['.', 'O', 'O']], [['golden', 'B-BUILDING', 'O'], ['lion', 'I-BUILDING', 'O'], ['inn', 'I-BUILDING', 'O'], [',', 'O', 'O'], ['wei', 'B-LOC', 'B-Q661619'], ['mouth', 'I-LOC', 'I-Q661619'], ['.', 'O', 'O']], [['mr', 'O', 'O'], ['armit', 'O', 'O'], ['«', 'O', 'O'], ['t', 'O', 'O'], ['.', 'O', 'O'], ['ad', 'O', 'O'], [',', 'O', 'O'], ['of', 'O', 'O'], ['gre', 'B-LOC', 'B-Q100'], ['>', 'I-LOC', 'I-Q100'], ['s', 'I-LOC', 'I-Q100'], [';', 'I-LOC', 'I-Q100'], ['nglia', 'I-LOC', 'I-Q100'], [',', 'O', 'O'], ['in', 'O', 'O'], ['this', 'O', 'O'], ['county', 'O', 'O'], ['.', 'O', 'O']]]

ner_trues = [[x[1] for x in x] for x in trues]
ner_preds = [[x[1] for x in x] for x in preds]

pred_mentions_sents = ner.aggregate_mentions(preds)
true_mentions_sents = ner.aggregate_mentions(trues)

### candidate selection

cand_select_method = 'deezy_match' # either perfect_match or deezy_match
mentions = list(set([mention['mention'] for sent in pred_mentions_sents for mention in sent]))

if mentions:
    cands = candidate_selection.select(mentions,cand_select_method)

### resolution
for s in range(len(pred_mentions_sents)):
    sent = pred_mentions_sents[s]
    for mention in sent:
        text_mention = mention['mention']
        start_offset = mention['start_offset']
        end_offset = mention['end_offset']
        # to be extended so that it can include multiple features and can consider sentence / document context
        res = linking.select(cands[text_mention],'most_popular')
        if res:
            link,score,other_cands = res
            for x in range(start_offset,end_offset+1):
                position_ner = preds[s][x][1][:2]
                preds[s][x][2] = position_ner+link
                preds[s][x].append(other_cands)


### Assessment of NER

#The SemEval’13 introduced four different ways to measure precision/recall/f1-score results based on the metrics defined by MUC.

#Strict: exact boundary surface string match and entity type;

#Exact: exact boundary match over the surface string, regardless of the type;

#Partial: partial boundary match over the surface string, regardless of the type;

#Type: some overlap between the system tagged entity and the gold annotation is required;

####
print ("\nNER Evaluation")

evaluator = eval.Evaluator(ner_trues, ner_preds, ['LOC', 'STREET','BUILDING'])
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

# Assessment of candidate selection
cand_sel_score = eval.eval_selection(true_mentions_sents,trues,preds)

print ('Only in {perc_cand}% of the cases we have retrieved the correct entity among the candidates.\n'.format(perc_cand=cand_sel_score*100))

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