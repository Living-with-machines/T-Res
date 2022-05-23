from geoparser import ranking
from utils import postprocess_data, linking, ner


class ELPipeline:
    def __init__(
        self,
        myner,
        myranker,
        mylinker,
        dataset,
    ):
        self.myner = myner
        self.myranker = myranker
        self.mylinker = mylinker
        self.dataset = dataset

    def run(self, sent, dataset=None, annotations=[], gold_positions=[], metadata=None):

        if self.myner.method == "rel":
            predicted_ents = [
                {
                    "wikidata_id": postprocess_data.match_wikipedia_to_wikidata(
                        pred[3]
                    ),
                    "ner_conf": pred[4],
                    "el_conf": pred[5],
                }
                for pred in annotations
            ]

            sentence_preds = []
            sentence_skys = []
            sentence_trues = []
            prev_ann = ""
            for token in gold_positions:
                start = token["start"]
                end = token["end"]
                word = token["word"]
                n, el, prev_ann = postprocess_data.match_ent(
                    annotations, start, end, prev_ann
                )
                sentence_preds.append([word, n, el])

        if self.myner.method == "lwm":
            gold_positions, predictions = self.myner.ner_predict(
                sent, annotations, self.dataset
            )
            sentence_preds = [
                [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
                for x in predictions
            ]
            sentence_trues = [
                [x["word"], x["entity"], x["link"], x["start"], x["end"]]
                for x in gold_positions
            ]
            sentence_skys = [
                [x["word"], x["entity"], "O", x["start"], x["end"]]
                for x in gold_positions
            ]

            # Filter by accepted labels:
            sentence_trues = [
                [x[0], x[1], "NIL", x[3], x[4]]
                if x[1] != "O" and x[1].lower() not in self.myner.filtering_labels()
                else x
                for x in sentence_trues
            ]

            pred_mentions_sent = self.myner.aggregate_mentions(sentence_preds)

            mentions = list(set([mention["mention"] for mention in pred_mentions_sent]))
            cands, self.already_collected_cands = self.myranker.run(mentions)

            predicted_ents = []
            for mention in pred_mentions_sent:
                text_mention = mention["mention"]
                start_offset = mention["start_offset"]
                end_offset = mention["end_offset"]
                start_char = mention["start_char"]
                end_char = mention["end_char"]

                mention_context = dict()
                mention_context["sentence"] = sent
                mention_context["mention"] = text_mention
                mention_context["mention_start"] = start_char
                mention_context["mention_end"] = end_char
                mention_context["metadata"] = metadata

                res = self.mylinker.run(cands[text_mention], mention_context)

                if res:
                    entity_candidate = cands[text_mention]
                    link, el_score, other_cands = res
                    mention["entity_candidate"] = entity_candidate
                    mention["wikidata_id"] = link
                    mention["el_score"] = el_score
                    predicted_ents.append(mention)

                    for x in range(start_offset, end_offset + 1):
                        position_ner = sentence_preds[x][1][:2]
                        sentence_preds[x][2] = position_ner + link
                        sentence_preds[x].append(other_cands)
                        true_label = sentence_trues[x][2].split("-")[-1]
                        if true_label in other_cands:
                            sentence_skys[x][2] = sentence_trues[x][2]

            # removing additional metadata not relevant for HIPE
            sentence_preds = [token[:3] for token in sentence_preds]
            sentence_trues = [token[:3] for token in sentence_trues]
            sentence_skys = [token[:3] for token in sentence_skys]

        output = {
            "predicted_ents": predicted_ents,
            "sentence_trues": sentence_trues,
            "sentence_preds": sentence_preds,
            "skyline": sentence_skys,
            "gold_positions": gold_positions,
        }

        return output
