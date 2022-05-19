import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_data


class Preprocessor:
    """
    The Preprocessor preprocesses, prepares, and formats the data for the experiments.
    """

    def __init__(
        self,
        dataset: str,
        data_path: str,
        results_path: str,
        dataset_df,
        myner,
        overwrite_processing=True,
        processed_data=dict(),
    ):
        """
        Arguments:
            dataset (str): dataset to process ("lwm" or "hipe").
            data_path (str): path to the processed data.
            results_path (str): path to the results of the experiments,
                initially empty (it is created if it does not exist).
            dataset_df (pd.DataFrame): initially empty dataframe
                where the resulting preprocessed dataset will be
                stored.
            myner (ner.Recogniser): a Recogniser object.
            overwrite_processing (bool): If True, do data processing,
                else load existing processing, if it exists.
            processed_data (dict): Dictionary where we'll keep the
                processed data for the experiments.
        """

        self.dataset = dataset
        self.data_path = data_path
        self.results_path = results_path
        self.myner = myner
        self.overwrite_processing = overwrite_processing
        self.dataset_df = dataset_df
        self.processed_data = processed_data

        # Load the dataset dataframe:
        self.dataset_df = pd.read_csv(
            self.data_path + self.dataset + "/linking_df_split.tsv",
            sep="\t",
        )

    def __str__(self):
        """
        Prints the data processor method name.
        """
        msg = "\nData processing in the " + self.dataset.upper() + " dataset."
        return msg

    def load_data(self):
        """
        Loads the processed data if exists.
        """
        return process_data.load_processed_data(self)

    def prepare_data(self):
        """
        Function that prepares the data for the experiments.

        Returns:
            dSentences (dict): dictionary in which we keep, for each article/sentence
                (expressed as e.g. "10732214_1", where "10732214" is the article_id
                and "1" is the order of the sentence in the article), the full original
                unprocessed sentence.
            dAnnotated (dict): dictionary in which we keep, for each article/sentence,
                an inner dictionary mapping the position of an annotated named entity (i.e.
                its start and end character, as a tuple, as the key) and another tuple as
                its value, which consists of: the type of named entity (such as LOC
                or BUILDING, the mention, and its annotated link), all extracted from
                the gold standard.
            dMetadata (dict): dictionary in which we keep, for each article/sentence,
                its metadata: place (of publication), year, ocr_quality_mean, ocr_quality_sd,
                publication_title, and publication_code.
            A JSON file in which we store the end-to-end resolution produced by REL
                using their API.
        """

        # If data is processed and overwrite is set to False, then do nothing,
        # otherwise process the data.

        if self.processed_data and self.overwrite_processing == False:
            print("\nData already postprocessed and loaded!\n")
            return self.processed_data

        # If data has not been processed, or overwrite is set to True, then:
        else:
            # Create the results directory if it does not exist:
            Path(self.results_path).mkdir(parents=True, exist_ok=True)

            # Prepare data per sentence:
            dAnnotated, dSentences, dMetadata = process_data.prepare_sents(
                self.dataset_df
            )

            # Print information on the Recogniser:
            print(self.myner)

            # Train the NER models if needed:
            print("*** Training the toponym recognition model...")
            self.myner.train()

            print("** Load NER pipeline!")
            self.myner.model, self.myner.pipe = self.myner.create_pipeline()
            accepted_labels = process_data.load_tagset(self.myner.filtering_labels)

            # =========================================
            # Parse with NER in the LwM way
            # =========================================

            print("\nPerform NER with our model:")
            output_lwm_ner = process_data.ner_and_process(
                dSentences, dAnnotated, self.myner, accepted_labels
            )
            dPreds = output_lwm_ner[0]
            dTrues = output_lwm_ner[1]
            dSkys = output_lwm_ner[2]
            gold_tokenization = output_lwm_ner[3]
            dMentionsPred = output_lwm_ner[4]
            dMentionsGold = output_lwm_ner[5]

            # ==========================================
            # Run REL end-to-end, as well
            # Note: Do not move the next block of code,
            # as REL relies on the tokenisation performed
            # by the previous method, so it needs to be
            # run after ther our method.
            # ==========================================

            Path(self.results_path + self.dataset).mkdir(parents=True, exist_ok=True)
            rel_end2end_path = (
                self.results_path + self.dataset + "/rel_e2d_from_api.json"
            )
            process_data.get_rel_from_api(dSentences, rel_end2end_path)
            print("\nPostprocess REL outputs:")
            dREL = process_data.postprocess_rel(
                rel_end2end_path, dSentences, gold_tokenization, accepted_labels
            )

            # ==========================================
            # Store postprocessed data
            # ==========================================

            self.processed_data = process_data.store_processed_data(
                dPreds,
                dTrues,
                dSkys,
                gold_tokenization,
                dSentences,
                dMetadata,
                dREL,
                dMentionsPred,
                dMentionsGold,
                self.data_path,
                self.dataset,
                self.myner.model_name,
                self.myner.filtering_labels,
            )
        return self.processed_data

    def create_df(self):
        dMentions = self.processed_data["dMentionsPred"]
        dGoldSt = self.processed_data["dMentionsGold"]
        dSentences = self.processed_data["dSentences"]
        dMetadata = self.processed_data["dMetadata"]
        self.processed_data["trues"]

        rows = []
        for sentence_id in dMentions:
            for mention in dMentions[sentence_id]:
                if mention:
                    article_id = sentence_id.split("_")[0]
                    sentence_pos = sentence_id.split("_")[1]
                    sentence = dSentences[sentence_id]
                    token_start = mention["start_offset"]
                    token_end = mention["end_offset"]
                    char_start = mention["start_char"]
                    char_end = mention["end_char"]
                    ner_score = round(mention["ner_score"], 3)
                    pred_mention = mention["mention"]
                    entity_type = mention["ner_label"]
                    place = dMetadata[sentence_id]["place"]
                    year = dMetadata[sentence_id]["year"]
                    publication = dMetadata[sentence_id]["publication_code"]
                    place_wqid = dMetadata[sentence_id]["place_wqid"]
                    # Match predicted mention with gold standard mention (will just be used for training):
                    max_tok_overlap = 0
                    gold_standard_link = "NIL"
                    gold_standard_ner = "O"
                    gold_mention = ""
                    for gs in dGoldSt[sentence_id]:
                        pred_token_range = range(token_start, token_end + 1)
                        gs_token_range = range(gs["start_offset"], gs["end_offset"] + 1)
                        overlap = len(list(set(pred_token_range) & set(gs_token_range)))
                        if overlap > max_tok_overlap:
                            max_tok_overlap = overlap
                            gold_mention = gs["mention"]
                            gold_standard_link = gs["entity_link"]
                            gold_standard_ner = gs["ner_label"]

                    rows.append(
                        [
                            sentence_id,
                            article_id,
                            sentence_pos,
                            sentence,
                            token_start,
                            token_end,
                            char_start,
                            char_end,
                            ner_score,
                            pred_mention,
                            entity_type,
                            place,
                            year,
                            publication,
                            place_wqid,
                            gold_mention,
                            gold_standard_link,
                            gold_standard_ner,
                        ]
                    )

        processed_df = pd.DataFrame(
            columns=[
                "sentence_id",
                "article_id",
                "sentence_pos",
                "sentence",
                "token_start",
                "token_end",
                "char_start",
                "char_end",
                "ner_score",
                "pred_mention",
                "pred_ner_label",
                "place",
                "year",
                "publication",
                "place_wqid",
                "gold_mention",
                "gold_entity_link",
                "gold_ner_label",
            ],
            data=rows,
        )

        output_path = (
            self.data_path
            + self.dataset
            + "/"
            + self.myner.model_name
            + "_"
            + self.myner.filtering_labels
        )

        # List of columns to merge (i.e. columns where we have indicated
        # out data splits), and "article_id", the columns on which we
        # will merge the data:
        keep_columns = [
            "article_id",
            "originalsplit",
            "traindevtest",
            "Ashton1860",
            "Dorchester1820",
            "Dorchester1830",
            "Dorchester1860",
            "Manchester1780",
            "Manchester1800",
            "Manchester1820",
            "Manchester1830",
            "Manchester1860",
            "Poole1860",
        ]

        # Add data splits from original dataframe:
        df = self.dataset_df[[c for c in keep_columns if c in self.dataset_df.columns]]

        # Convert article_id to string (it's read as an int):
        df = df.assign(article_id=lambda d: d["article_id"].astype(str))
        processed_df = processed_df.assign(
            article_id=lambda d: d["article_id"].astype(str)
        )
        processed_df = pd.merge(processed_df, df, on=["article_id"], how="left")

        # Store mentions dataframe:
        processed_df.to_csv(output_path + "_linking_df_mentions.tsv", sep="\t")

        return processed_df
