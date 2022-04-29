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

    def run(self, sent, dataset=None, annotations=[], gold_positions=[], metadata=None):

        if self.ner_model_id == "rel":
            predicted_tags = linking.rel_end_to_end(sent)
            predicted_ents = [
                {
                    "wikidata_id": process_data.match_wikipedia_to_wikidata(pred[3]),
                    "ner_conf": pred[4],
                    "el_conf": pred[5],
                }
                for pred in predicted_tags
            ]

            sentence_preds = []
            sentence_skys = []
            sentence_trues = []
            prev_ann = ""
            for token in gold_positions:
                start = token["start"]
                end = token["end"]
                word = token["word"]
                n, el, prev_ann = process_data.match_ent(predicted_tags, start, end, prev_ann)
                sentence_preds.append([word, n, el])

        if self.ner_model_id == "lwm":
            gold_positions, predictions = ner.ner_predict(
                sent, annotations, self.ner_pipe, dataset
            )
            sentence_preds = [
                [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
                for x in predictions
            ]
            sentence_trues = [
                [x["word"], x["entity"], x["link"], x["start"], x["end"]] for x in gold_positions
            ]
            sentence_skys = [
                [x["word"], x["entity"], "O", x["start"], x["end"]] for x in gold_positions
            ]
            # Filter by accepted labels:
            sentence_trues = [
                [x[0], x[1], "NIL", x[3], x[4], x[5]]
                if x[1] != "O" and x[1].lower() not in self.accepted_labels
                else x
                for x in sentence_trues
            ]

            pred_mentions_sent = ner.aggregate_mentions(sentence_preds, self.accepted_labels)

            mentions = list(set([mention["mention"] for mention in pred_mentions_sent]))
            cands, self.already_collected_cands = candidate_selection.select(
                mentions,
                self.cand_select_method,
                self.myranker,
                self.already_collected_cands,
            )
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
                self.mylinker["mention_context"] = mention_context
                self.mylinker["metadata"] = metadata

                # TO DO: FIND CORRECT PLACE OF PUBLICATION FOR HIPE:
                if dataset == "hipe":
                    self.mylinker["metadata"]["place"] = "New York"

                res = linking.select(cands[text_mention], self.top_res_method, self.mylinker)

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
