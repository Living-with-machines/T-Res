from utils import ner, candidate_selection, linking
from sklearn.model_selection import train_test_split
from transformers import pipeline
import pandas as pd
import pathlib
import tqdm

# Dataset:
dataset = "lwm"

# Path to NER Model:
ner_model = "/resources/develop/mcollardanuy/toponym-resolution/outputs/models/lwm-ner.model"

# Path to test dataframe:
df = pd.read_csv("/resources/develop/mcollardanuy/toponym-resolution/outputs/data/linking_lwm_df_test.tsv", sep="\t")

# Split test set into dev and test set (by article, not sentence):
dev_ids, test_ids = train_test_split(df.article_id.unique(), test_size=0.5, random_state=42)
dev = df[df["article_id"].isin(dev_ids)]
test = df[df["article_id"].isin(test_ids)]

ner_pipe = pipeline("ner", model=ner_model)
cand_select_method = 'perfect_match' # either perfect_match or deezy_match

dAnnotated, dSentences = ner.format_for_ner(dev)

preds = []
trues = []
dPreds = dict()
dTrues = dict()
for sent_id in tqdm.tqdm(dSentences.keys()):
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
    for mention in pred_mentions_sents:
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

    dPreds[sent_id] = sentence_preds
    dTrues[sent_id] = sentence_trues


# Storing results for evaluation using the CLEF-HIPE scorer
def store_results_hipe(dataset, dataresults, dresults):
    """
    Store results in the right format to be used by the CLEF-HIPE
    scorer: https://github.com/impresso/CLEF-HIPE-2020-scorer.

    Assuming the CLEF-HIPE scorer is stored in ../CLEF-HIPE-2020-scorer/,
    run scorer as follows:
    For NER:
    > python ../CLEF-HIPE-2020-scorer/clef_evaluation.py --ref outputs/results/lwm-true_bundle2_en_1.tsv --pred outputs/results/lwm-pred_bundle2_en_1.tsv --task nerc_coarse --outdir outputs/results/
    For EL:
    > python ../CLEF-HIPE-2020-scorer/clef_evaluation.py --ref outputs/results/lwm-true_bundle2_en_1.tsv --pred outputs/results/lwm-pred_bundle2_en_1.tsv --task nel --outdir outputs/results/
    """
    pathlib.Path("outputs/results/").mkdir(parents=True, exist_ok=True)
    # Bundle 2 associated tasks: NERC-coarse and NEL
    with open("outputs/results/" + dataset + "-" + dataresults + "_bundle2_en_1.tsv", "w") as fw:
        fw.write("TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tNE-FINE-LIT\tNE-FINE-METO\tNE-FINE-COMP\tNE-NESTED\tNEL-LIT\tNEL-METO\tMISC\n")
        for sent_id in dresults:
            fw.write("# sentence_id = " + sent_id + "\n")
            for t in dresults[sent_id]:
                elink = t[2]
                if t[2].startswith("B-"):
                    elink = t[2].replace("B-", "")
                elif t[2].startswith("I-"):
                    elink = t[2].replace("I-", "")
                elif t[1] != "O":
                    elink = "NIL"
                fw.write(t[0] + "\t" + t[1] + "\t0\tO\tO\tO\tO\t" + elink + "\tO\tO\n")
            fw.write("\n")
    
store_results_hipe(dataset, "pred", dPreds)
store_results_hipe(dataset, "true", dTrues)