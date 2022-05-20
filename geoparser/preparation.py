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

            # -------------------------------------------
            # Parse with NER in the LwM way
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

            # -------------------------------------------
            # Run REL end-to-end, as well
            # Note: Do not move the next block of code,
            # as REL relies on the tokenisation performed
            # by the previous method, so it needs to be
            # run after ther our method.
            Path(self.results_path + self.dataset).mkdir(parents=True, exist_ok=True)
            rel_end2end_path = (
                self.results_path + self.dataset + "/rel_e2d_from_api.json"
            )
            process_data.get_rel_from_api(dSentences, rel_end2end_path)
            print("\nPostprocess REL outputs:")
            dREL = process_data.postprocess_rel(
                rel_end2end_path, dSentences, gold_tokenization, accepted_labels
            )

            # -------------------------------------------
            # Store postprocessed data
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

        # -------------------------------------------
        # Store results in the CLEF-HIPE scorer-required format
        hipe_scorer_results_path = self.results_path + self.dataset + "/"
        scenario_name = (
            "ner_" + self.myner.model_name + "_" + self.myner.filtering_labels + "_"
        )

        # Find article ids of the test set (original split, for NER):
        ner_all = self.dataset_df
        ner_test_articles = list(
            ner_all[ner_all["originalsplit"] == "test"].article_id.unique()
        )
        ner_test_articles = [str(art) for art in ner_test_articles]

        # Store predictions results formatted for CLEF-HIPE scorer:
        process_data.store_for_scorer(
            hipe_scorer_results_path,
            scenario_name + "preds",
            self.processed_data["preds"],
            ner_test_articles,
        )

        # Store gold standard results formatted for CLEF-HIPE scorer:
        process_data.store_for_scorer(
            hipe_scorer_results_path,
            scenario_name + "trues",
            self.processed_data["trues"],
            ner_test_articles,
        )

        # Store REL results formatted for CLEF-HIPE scorer:
        process_data.store_for_scorer(
            hipe_scorer_results_path,
            scenario_name + "rel",
            self.processed_data["dREL"],
            ner_test_articles,
        )

        # Create a mention-based dataframe for the linking experiments:
        processed_df = process_data.create_mentions_df(self)
        self.processed_data["processed_df"] = processed_df

        return self.processed_data
