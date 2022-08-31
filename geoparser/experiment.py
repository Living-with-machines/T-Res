import os
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_data, training


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
        rel_experiments=False,
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
            rel_experiments (bool): If True, run the REL experiments,
                else skip them.
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
        self.rel_experiments = rel_experiments

        # Load the dataset as a dataframe:
        dataset_path = os.path.join(
            self.data_path, self.dataset, "linking_df_split.tsv"
        )

        if Path(dataset_path).exists():
            self.dataset_df = pd.read_csv(
                dataset_path,
                sep="\t",
            )
        else:
            sys.exit(
                "\nError: The dataset has not been created, you should first run the data_processing.py script.\n"
            )

    def __str__(self):
        """
        Prints the characteristics of the experiment.
        """
        msg = "\n>>> Experiment\n"
        msg += "    * Dataset: {0}\n".format(self.dataset.upper())
        msg += "    * Overwrite processing: {0}\n".format(self.overwrite_processing)
        msg += "    * Experiments on: {0}\n".format(self.test_split)
        msg += "    * Run end-to-end REL experiments: {0}".format(self.rel_experiments)
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
        # Coherence checks:
        # Some scenarios do not make sense. Warn and exit:
        if self.myranker.method not in [
            "perfectmatch",
            "partialmatch",
            "levenshtein",
            "deezymatch",
            "relcs",
        ]:
            print(
                "\n!!! Coherence check failed. "
                "This is because the candidate ranking method does not exist.\n"
            )
            sys.exit(0)
        if self.dataset == "hipe" and self.myner.training_tagset != "coarse":
            print(
                "\n!!! Coherence check failed. "
                "HIPE should be run with the coarse tagset.\n"
            )
            sys.exit(0)
        if self.myranker.method == "relcs" and not self.mylinker.method == "reldisamb":
            print(
                "\n!!! Coherence check failed. "
                "The REL candidate selection method only works with the reldisamb.\n"
                "linking method.\n"
            )
            sys.exit(0)

        # ----------------------------------
        # If data is processed and overwrite is set to False, then do nothing,
        # otherwise process the data.
        if self.processed_data and self.overwrite_processing == False:
            print("\nData already postprocessed and loaded!\n")
            return self.processed_data

        # ----------------------------------
        # If data has not been processed, or overwrite is set to True, then:

        # Create the results directory if it does not exist:
        Path(self.results_path).mkdir(parents=True, exist_ok=True)

        # Prepare data per sentence:
        dAnnotated, dSentences, dMetadata = process_data.prepare_sents(self.dataset_df)

        # -----------------------------------------
        # NER training and creating pipeline:
        # Train the NER models if needed:
        self.myner.train()
        # Load the NER pipeline:
        self.myner.model, self.myner.pipe = self.myner.create_pipeline()

        # -----------------------------------------
        # Ranker loading resources and training a model:
        # Load the resources:
        self.myranker.mentions_to_wikidata = self.myranker.load_resources()
        # Train a DeezyMatch model if needed:
        self.myranker.train()

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
        print("\n* Perform candidate ranking:")
        dCandidates = dict()
        # Obtain candidates per sentence:
        for sentence_id in tqdm(dMentionsPred):
            if self.myranker.method == "relcs":
                # REL match is performed in in a later stage, during the
                # linking. Therefore at this point we just return an empty
                # dictionary.
                dCandidates[sentence_id] = dict()
            else:
                pred_mentions_sent = dMentionsPred[sentence_id]
                (
                    wk_cands,
                    self.myranker.already_collected_cands,
                ) = self.myranker.find_candidates(pred_mentions_sent)
                dCandidates[sentence_id] = wk_cands

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
            dMentionsPred,
            dMentionsGold,
            dCandidates,
        )

        # -------------------------------------------
        # Store results in the CLEF-HIPE scorer-required format
        process_data.store_results(
            self, task="ner", how_split="originalsplit", which_split="test"
        )

        return self.processed_data

    def linking_experiments(self):

        # Candidates:
        cand_selection = "relcs" if self.myranker.method == "relcs" else "lwmcs"

        # Create a mention-based dataframe for the linking experiments:
        processed_df = process_data.create_mentions_df(self)
        self.processed_data["processed_df"] = processed_df

        # Load linking resources:
        self.mylinker.linking_resources = self.mylinker.load_resources()

        # Experiments data splits:
        if self.test_split == "dev":
            list_test_splits = ["withouttest"]
        if self.test_split == "test":
            if self.dataset == "hipe":
                list_test_splits = ["originalsplit"]
            elif self.dataset == "lwm":
                list_test_splits = [
                    "originalsplit",
                    # "Ashton1860",
                    # "Dorchester1820",
                    # "Dorchester1830",
                    # "Dorchester1860",
                    # "Manchester1780",
                    # "Manchester1800",
                    # "Manchester1820",
                    # "Manchester1830",
                    # "Manchester1860",
                    # "Poole1860",
                ]

        train_original = pd.DataFrame()
        train_processed = pd.DataFrame()
        if self.mylinker.rel_params.get("training_data") == "lwm":
            train_original, train_processed = training.load_training_lwm_data(self)

        # ------------------------------------------
        # Iterate over each linking experiments, each will have its own
        # results file:
        for split in list_test_splits:

            processed_df = self.processed_data["processed_df"]
            original_df = self.dataset_df

            dev_processed = processed_df[processed_df[split] == "dev"]
            test_processed = processed_df[processed_df[split] == "test"]
            dev_original = original_df[original_df[split] == "dev"]
            test_original = original_df[original_df[split] == "test"]

            # Get ids of articles in each split:
            test_article_ids = list(test_original.article_id.astype(str))

            # Get the experiment name:
            cand_approach = self.myranker.method
            if self.myranker.method == "deezymatch":
                cand_approach += "+" + str(
                    self.myranker.deezy_parameters["num_candidates"]
                )
                cand_approach += "+" + str(
                    self.myranker.deezy_parameters["selection_threshold"]
                )
            link_approach = self.mylinker.method
            if self.mylinker.method == "reldisamb":
                link_approach += "+" + str(self.mylinker.rel_params["ranking"])
            experiment_name = self.mylinker.rel_params["training_data"]
            if self.mylinker.rel_params["training_data"] == "lwm":
                experiment_name += "_" + cand_approach
                experiment_name += "_" + link_approach
                experiment_name += "_" + split
            if self.mylinker.rel_params["training_data"] == "aida":
                experiment_name += "_" + cand_approach
                experiment_name += "_" + link_approach

            # If method is supervised, train and store model:
            if "reldisamb" in self.mylinker.method:
                self.mylinker.rel_params = self.mylinker.perform_training(
                    train_original,
                    train_processed,
                    dev_original,
                    dev_processed,
                    experiment_name,
                    cand_selection,
                )

            # Resolve according to method:
            test_df = self.mylinker.perform_linking(
                test_processed,
                test_original,
                experiment_name,
                cand_selection,
            )

            # Prepare data for scorer:
            self.processed_data = process_data.prepare_storing_links(
                self.processed_data, test_article_ids, test_df
            )

            # Store linking results:
            process_data.store_results(
                self,
                task="linking",
                how_split=split,
                which_split=self.test_split,
            )

        # -----------------------------------------------
        # Run end-to-end REL experiments:
        if self.rel_experiments == True:
            from utils import rel_e2e

            rel_e2e.run_rel_experiments(self)
