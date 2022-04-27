from utils import candidate_selection, linking, ner, process_data


class ELPipeline:
    def __init__(
        self,
        ner_model_id,
        cand_select_method,
        top_res_method,
        myranker,
        mylinker,
        accepted_labels,
        ner_pipe,
    ):
        self.ner_model_id = ner_model_id
        self.cand_select_method = cand_select_method
        self.top_res_method = top_res_method
        self.myranker = myranker
        self.mylinker = mylinker
        self.accepted_labels = accepted_labels
        self.already_collected_cands = {}
        self.ner_pipe = ner_pipe

    def run(self, sent):
        if self.ner_model_id == "rel":
            pred_ents = linking.rel_end_to_end(sent)
            pred_ents = [
                {
                    "wikidata_id": process_data.match_wikipedia_to_wikidata(pred[3]),
                    "ner_conf": pred[4],
                    "el_conf": pred[5],
                }
                for pred in pred_ents
            ]
        if self.ner_model_id == "lwm":
            gold_standard, predictions = ner.ner_predict(sent, [], self.ner_pipe, "lwm")
            sentence_preds = [
                [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
                for x in predictions
            ]
            pred_mentions_sent = ner.aggregate_mentions(sentence_preds, self.accepted_labels)

            mentions = list(set([mention["mention"] for mention in pred_mentions_sent]))
            cands, self.already_collected_cands = candidate_selection.select(
                mentions,
                self.cand_select_method,
                self.myranker,
                self.already_collected_cands,
            )
            pred_ents = []
            for mention in pred_mentions_sent:
                text_mention = mention["mention"]
                res = linking.select(cands[text_mention], self.top_res_method, self.mylinker)
                if res:
                    entity_candidate = cands[text_mention]
                    link, el_score, other_cands = res
                    mention["entity_candidate"] = entity_candidate
                    mention["wikidata_id"] = link
                    mention["el_score"] = el_score
                    pred_ents.append(mention)

        return pred_ents
