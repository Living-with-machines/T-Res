import json
import operator
from utils import ner
from sklearn.metrics import precision_recall_fscore_support

# Path to NER Model:
ner_model = "/resources/develop/mcollardanuy/toponym-resolution/outputs/models/lwm-ner.model"

wikidata_path = '/resources/wikidata/'
with open(wikidata_path + 'mentions_to_wikidata.json', 'r') as f:
    mentions_to_wikidata = json.load(f)

preds = [[('mr', 'O', 'O'), ('.', 'O', 'O'), ('oldham', 'O', 'O'), (',', 'O', 'O'), ('ol', 'O', 'O'), ('London', 'B-LOC', 'O'), (',', 'O', 'O'), ('in', 'O', 'O'), ('stafford', 'B-LOC', 'O'), ('(', 'I-LOC', 'O'), ('hire', 'I-LOC', 'O'), (',', 'O', 'O'), ('to', 'O', 'O'), ('mils', 'O', 'O'), ('oldlcaa', 'O', 'O'), (',', 'O', 'O'), ('lie', 'O', 'O'), ('of', 'O', 'O'), ('this', 'O', 'O'), ('town', 'O', 'O'), ('.', 'O', 'O')], [('glo', 'O', 'O'), ('uc', 'O', 'O'), ('est', 'O', 'O'), ('ers', 'O', 'O'), ('h1', 'O', 'O'), ('re', 'O', 'O'), ('.', 'O', 'O')], [('golden', 'B-BUILDING', 'O'), ('lion', 'I-BUILDING', 'O'), ('inn', 'O', 'O'), (',', 'O', 'O'), ('wei', 'B-LOC', 'O'), ('mouth', 'I-LOC', 'O'), ('.', 'O', 'O')], [('mr', 'O', 'O'), ('armit', 'O', 'O'), ('«', 'O', 'O'), ('t', 'O', 'O'), ('.', 'O', 'O'), ('ad', 'O', 'O'), (',', 'O', 'O'), ('of', 'O', 'O'), ('gre', 'B-LOC', 'O'), ('>', 'I-LOC', 'O'), ('s', 'I-LOC', 'O'), (';', 'I-LOC', 'O'), ('nglia', 'I-LOC', 'O'), (',', 'O', 'O'), ('in', 'O', 'O'), ('this', 'O', 'O'), ('county', 'O', 'O'), ('.', 'O', 'O')]]
trues = [[('mr', 'O', 'O'), ('.', 'O', 'O'), ('oldham', 'O', 'O'), (',', 'O', 'O'), ('ol', 'O', 'O'), ('London', 'B-LOC', 'Q84'), (',', 'O', 'O'), ('in', 'O', 'O'), ('stafford', 'B-LOC', 'O'), ('(', 'I-LOC', 'O'), ('hire', 'I-LOC', 'O'), (',', 'O', 'O'), ('to', 'O', 'O'), ('mils', 'O', 'O'), ('oldlcaa', 'O', 'O'), (',', 'O', 'O'), ('lie', 'O', 'O'), ('of', 'O', 'O'), ('this', 'O', 'O'), ('town', 'O', 'O'), ('.', 'O', 'O')], [('glo', 'B-LOC', 'Q23165'), ('uc', 'I-LOC', 'Q23165'), ('est', 'I-LOC', 'Q23165'), ('ers', 'I-LOC', 'Q23165'), ('h1', 'I-LOC', 'Q23165'), ('re', 'I-LOC', 'Q23165'), ('.', 'O', 'O')], [('golden', 'B-BUILDING', 'O'), ('lion', 'I-BUILDING', 'O'), ('inn', 'I-BUILDING', 'O'), (',', 'O', 'O'), ('wei', 'B-LOC', 'Q661619'), ('mouth', 'I-LOC', 'Q661619'), ('.', 'O', 'O')], [('mr', 'O', 'O'), ('armit', 'O', 'O'), ('«', 'O', 'O'), ('t', 'O', 'O'), ('.', 'O', 'O'), ('ad', 'O', 'O'), (',', 'O', 'O'), ('of', 'O', 'O'), ('gre', 'B-LOC', 'O'), ('>', 'I-LOC', 'O'), ('s', 'I-LOC', 'O'), (';', 'I-LOC', 'O'), ('nglia', 'I-LOC', 'O'), (',', 'O', 'O'), ('in', 'O', 'O'), ('this', 'O', 'O'), ('county', 'O', 'O'), ('.', 'O', 'O')]]

evaluator = ner.Evaluator(trues, preds, ['LOC', 'STREET','BUILDING'])
results, results_agg = evaluator.evaluate()

print ("LOC")
for res,scores in results_agg["LOC"].items():
    print (res,"p:",scores["precision"],"r:",scores["recall"])

print ("BUILDING")
for res,scores in results_agg["BUILDING"].items():
    print (res,"p:",scores["precision"],"r:",scores["recall"])

print ("STREET")
for res,scores in results_agg["STREET"].items():
    print (res,"p:",scores["precision"],"r:",scores["recall"])

pred_mentions_sents = ner.aggregate_mentions(preds)

# candidate selection and resolution step
for sent in pred_mentions_sents:
    for mention in sent:
        text_mention = mention['mention']
        # candidate selection step (make this as given mention returns list of candidates)
        if text_mention in mentions_to_wikidata:
            # make this as given list of candidates return link
            cands = sorted(mentions_to_wikidata[text_mention].items(), key=operator.itemgetter(1),reverse=True)
            link,score = cands[0]
            print (text_mention,'~~>',link, score)