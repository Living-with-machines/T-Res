import os
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_data


class Experiment:
    """
    The Experiment class processes, prepares, and formats the data for the experiments.
    """

    def __init__(
        self,
        dataset: str,
        data_path: str,
        results_path: str,
        dataset_df,
        myner,
        myranker,
        mylinker,
        overwrite_processing=True,
        processed_data=dict(),
        test_split="",
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
            myner (recogniser.Recogniser): a Recogniser object.
            myranker (ranking.Ranker): a Ranker object.
            mylinker (linking.Linker): a Linker object.
            overwrite_processing (bool): If True, do data processing,
                else load existing processing, if it exists.
            processed_data (dict): Dictionary where we'll keep the
                processed data for the experiments.
            test split (str): the split (train/dev/test) for the
                linking experiment.
        """

        self.dataset = dataset
        self.data_path = data_path
        self.results_path = results_path
        self.myner = myner
        self.myranker = myranker
        self.mylinker = mylinker
        self.overwrite_processing = overwrite_processing
        self.dataset_df = dataset_df
        self.processed_data = processed_data
        self.test_split = test_split

        # Load the dataset as a dataframe:
        self.dataset_df = pd.read_csv(
            self.data_path + self.dataset + "/linking_df_split.tsv",
            sep="\t",
        )

    def __str__(self):
        """
        Prints the characteristics of the experiment.
        """
        msg = "\nData processing in the " + self.dataset.upper() + " dataset."
        msg += "\n* Overwrite processing: " + str(self.overwrite_processing)
        msg += "\n* Experiments run on the >>> " + self.test_split + " <<< set.\n"
        return msg

    def load_data(self):
        """
        Loads the already processed data if exists.
        """
        return process_data.load_processed_data(self)

    def prepare_data(self):
        """
        Function that prepares the data for the experiments.

        Returns:
            self.processed_data (dict): a dictionary which stores the different
                processed data (predicted mentions, gold standard, REL end-to-end
                processing, candidates), which will be used later for linking.
            A JSON file in which we store the end-to-end resolution produced by REL
                using their API.
        """

        # ----------------------------------
        # Coherence check:
        # Some scenarios do not make sense. Warn and exit:
        if self.dataset == "hipe" and self.myner.training_tagset == "fine":
            print(
                """\n!!! Coherence check failed. This is due to:
                * HIPE should neither be filtered by type of label nor allow processing 
                  with fine-graned location types, because it was not designed for that.\n"""
            )
            sys.exit(0)

        # ----------------------------------
        # If data is processed and overwrite is set to False, then do nothing,
        # otherwise process the data.
        if self.processed_data and self.overwrite_processing == False:
            print("\nData already postprocessed and loaded!\n")

        # ----------------------------------
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

            # -------------------------------------------
            # Parse with NER in the LwM way
            print("\nPerform NER with our model:")
            output_lwm_ner = process_data.ner_and_process(
                dSentences, dAnnotated, self.myner
            )
            dPreds = output_lwm_ner[0]
            dTrues = output_lwm_ner[1]
            dSkys = output_lwm_ner[2]
            gold_tokenization = output_lwm_ner[3]
            dMentionsPred = output_lwm_ner[4]
            dMentionsGold = output_lwm_ner[5]

            # -------------------------------------------
            # Perform candidate ranking:
            # Load ranking resources
            print("\n* Load ranking resources:")
            self.myranker.mentions_to_wikidata = self.myranker.load_resources()
            print("\n* Perform candidate ranking:")
            # Obtain candidates per sentence:
            dCandidates = dict()
            for sentence_id in tqdm(dMentionsPred):
                pred_mentions_sent = dMentionsPred[sentence_id]
                (
                    wk_cands,
                    self.myranker.already_collected_cands,
                ) = self.myranker.find_candidates(pred_mentions_sent)
                dCandidates[sentence_id] = wk_cands

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
                rel_end2end_path, dSentences, gold_tokenization
            )

            # -------------------------------------------
            # Store temporary postprocessed data
            self.processed_data = process_data.store_processed_data(
                self,
                dPreds,
                dTrues,
                dSkys,
                gold_tokenization,
                dSentences,
                dMetadata,
                dREL,
                dMentionsPred,
                dMentionsGold,
                dCandidates,
            )

        # -------------------------------------------
        # Store results in the CLEF-HIPE scorer-required format
        process_data.store_results(
            self, task="ner", how_split="originalsplit", which_split="test"
        )

        # Create a mention-based dataframe for the linking experiments:
        processed_df = process_data.create_mentions_df(self)
        self.processed_data["processed_df"] = processed_df

        return self.processed_data

    def linking_experiments(self):

        # Linker load resources:
        print("\n* Load linking resources...")
        self.mylinker.linking_resources = self.mylinker.load_resources()
        print("... resources loaded, linking in progress!\n")

        list_test_splits = []
        if self.test_split == "dev":
            # We use the original split for developing the code:
            list_test_splits = ["originalsplit"]

        if self.test_split == "test":
            # N-cross validation (we use the originalsplit for developing the
            # code, when running the code for real we'll uncomment the following
            # block of code):
            if self.dataset == "hipe":
                list_test_splits += ["originalsplit", "traindevtest"]
            elif self.dataset == "lwm":
                list_test_splits += [
                    "originalsplit",
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

        # Iterate over each linking experiments, each will have its own
        # results file:
        all_df = self.dataset_df
        for split in list_test_splits:
            # Get ids of articles in each split:
            all_test = list(
                all_df[all_df[split] == self.test_split].article_id.astype(str)
            )

            # Split processed df according to split:
            processed_df = self.processed_data["processed_df"]
            train_df = processed_df[processed_df[split] == "train"]
            test_df = processed_df[processed_df[split] == self.test_split]

            # Train according to method:
            self.mylinker.perform_training(train_df)

            # Resolve according to method:
            test_df = self.mylinker.perform_linking(test_df)

            # Prepare data for scorer:
            self.processed_data = process_data.prepare_storing_links(
                self.processed_data, all_test, test_df
            )

            # Store linking results:
            process_data.store_results(
                self,
                task="linking",
                how_split=split,
                which_split=self.test_split,
            )
