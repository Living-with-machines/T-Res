import os
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_data, rel_utils


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
        Initialises an Experiment object.

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
                "\nError: The dataset has not been created, you should first run the prepare_data.py script.\n"
            )

    def __str__(self):
        """
        Prints information about the experiment.
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
        ]:
            print(
                "\n!!! Coherence check failed. "
                "This is because the candidate ranking method does not exist.\n"
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
        """
        Prepares the data for the linking experiments, creating a mention-based
        dataframe. It produces tsv files in the format required by the HIPE
        scorer, ready to be evaluated.
        """
        # Create a mention-based dataframe for the linking experiments:
        processed_df = process_data.create_mentions_df(self)
        self.processed_data["processed_df"] = processed_df

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

        # ------------------------------------------
        # Iterate over each linking experiments, each will have its own
        # results file:
        for split in list_test_splits:
            original_df = self.dataset_df
            processed_df = self.processed_data["processed_df"]

            test_original = original_df[original_df[split] == "test"]
            test_processed = processed_df[processed_df[split] == "test"]

            # Get ids of articles in each split:
            test_article_ids = list(test_original.article_id.astype(str))

            # Train a linking model if needed (it requires myranker to generate potential
            # candidates to the training set):
            linking_model = self.mylinker.train_load_model(self.myranker, split=split)

            # Dictionary of sentences:
            # {k1 : {k2 : v}}, where k1 is article id, k2 is
            # sentence pos, and v is the sentence text.
            nested_sentences_dict = dict()
            for key, val in self.processed_data["dSentences"].items():
                key1, key2 = key.split("_")
                key2 = int(key2)
                if key1 in nested_sentences_dict:
                    nested_sentences_dict[key1][key2] = val
                else:
                    nested_sentences_dict[key1] = {key2: val}

            # Predict:
            print("Process data into sentences.")
            to_append = []
            mentions_dataset = dict()
            all_cands = dict()
            for i, row in tqdm(test_processed.iterrows()):
                prediction = dict()
                mention_data = row.to_dict()
                sentence_id = mention_data["sentence_id"]
                article_id = mention_data["article_id"]
                prediction["mention"] = mention_data["pred_mention"]
                # Generate left-hand context:
                left_context = ""
                sent_idx = int(mention_data["sentence_pos"])
                if sent_idx - 1 in nested_sentences_dict[article_id]:
                    left_context = nested_sentences_dict[article_id][sent_idx - 1]
                # Generate right-hand context:
                right_context = ""
                if sent_idx + 1 in nested_sentences_dict[article_id]:
                    right_context = nested_sentences_dict[article_id][sent_idx + 1]
                prediction["context"] = [left_context, right_context]
                prediction["candidates"] = mention_data["candidates"]
                prediction["gold"] = ["NONE"]
                prediction["ner_score"] = mention_data["ner_score"]
                prediction["pos"] = mention_data["char_start"]
                prediction["sent_idx"] = sent_idx
                prediction["end_pos"] = mention_data["char_end"]
                prediction["ngram"] = mention_data["pred_mention"]
                prediction["conf_md"] = mention_data["ner_score"]
                prediction["tag"] = mention_data["pred_ner_label"]
                prediction["sentence"] = mention_data["sentence"]
                prediction["place"] = mention_data["place"]
                prediction["place_wqid"] = mention_data["place_wqid"]
                if self.mylinker.method == "reldisamb":
                    if (
                        self.mylinker.rel_params["without_microtoponyms"]
                        and mention_data["pred_ner_label"] != "LOC"
                    ):
                        prediction["candidates"] = dict()
                if sentence_id in mentions_dataset:
                    mentions_dataset[sentence_id].append(prediction)
                else:
                    mentions_dataset[sentence_id] = [prediction]
                all_cands.update({prediction["mention"]: prediction["candidates"]})

            if self.mylinker.method == "reldisamb":
                rel_resolved = dict()
                for sentence_id in mentions_dataset:
                    article_dataset = {sentence_id: mentions_dataset[sentence_id]}
                    article_dataset = rel_utils.rank_candidates(
                        article_dataset,
                        all_cands,
                        self.mylinker.linking_resources["mentions_to_wikidata"],
                    )
                    if self.mylinker.rel_params["with_publication"]:
                        # If "publ", add an artificial publication entry:
                        article_dataset = rel_utils.add_publication(article_dataset)
                    predicted = linking_model.predict(article_dataset)
                    if self.mylinker.rel_params["with_publication"]:
                        # ... and if "publ", now remove the artificial publication entry!
                        predicted[sentence_id].pop()
                    for i in range(len(predicted[sentence_id])):
                        combined_mention = article_dataset[sentence_id][i]
                        combined_mention["prediction"] = predicted[sentence_id][i][
                            "prediction"
                        ]
                        combined_mention["ed_score"] = predicted[sentence_id][i][
                            "conf_ed"
                        ]
                        if sentence_id in rel_resolved:
                            rel_resolved[sentence_id].append(combined_mention)
                        else:
                            rel_resolved[sentence_id] = [combined_mention]
                    mentions_dataset[sentence_id] = rel_resolved[sentence_id]

            for i, row in tqdm(test_processed.iterrows()):
                prediction = dict()
                for mention in mentions_dataset[row["sentence_id"]]:
                    if (
                        int(mention["pos"]) == int(row["char_start"])
                        and int(mention["sent_idx"]) == int(row["sentence_pos"])
                        and mention["mention"] == row["pred_mention"]
                    ):
                        prediction = mention

                        if self.mylinker.method in ["mostpopular", "bydistance"]:
                            # Run entity linking per mention:
                            selected_cand = self.mylinker.run(
                                {
                                    "candidates": prediction["candidates"],
                                    "place_wqid": prediction["place_wqid"],
                                }
                            )
                            prediction["prediction"] = selected_cand[0]
                            prediction["ed_score"] = round(selected_cand[1], 3)

                to_append.append(
                    [
                        prediction["prediction"],
                        round(prediction["ed_score"], 3),
                    ]
                )

            test_df = test_processed.copy()
            test_df[["pred_wqid", "ed_score"]] = to_append

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
