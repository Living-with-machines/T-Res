import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import linking, ranking, recogniser
from utils import process_data, rel_utils


class Experiment:
    """
    A class to represent an an entity linking experiment using NER,
    candidate ranking, and linking methods.

    Arguments:
        dataset ("lwm", "hipe"): The dataset to use for the
            experiment, must be set to either ``"lwm"`` or ``"hipe"``.
        data_path (str): The path to the dataset directory (with processed
            data).
        results_path (str): The path to the directory where the results will
            be stored. If it does not exist, it will be created.
        dataset_df (pandas.DataFrame): The dataframe representing the
            resulting, preprocessed, dataset.
        myner (recogniser.Recogniser): An instance of the NER model to use.
        myranker (ranking.Ranker): An instance of the candidate ranking model
            to use.
        mylinker (linking.Linker): An instance of the linking model to use.
        overwrite_processing (bool, optional): Whether to overwrite the
            processed data if it already exists (default is ``True``).
        processed_data (dict, optional): A dictionary to store the processed
            data (default is an empty dictionary).
        test split (str, optional): The data split to use for testing (train/
            dev/test, default is an empty string).
        rel_experiments (bool, optional): Whether to run end-to-end REL
            experiments (default is ``False``).
    """

    def __init__(
        self,
        dataset: Literal["lwm", "hipe"],
        data_path: str,
        results_path: str,
        dataset_df: pd.DataFrame,
        myner: recogniser.Recogniser,
        myranker: ranking.Ranker,
        mylinker: linking.Linker,
        overwrite_processing: Optional[bool] = True,
        processed_data: Optional[dict] = dict(),
        test_split: Optional[str] = "",
        rel_experiments: Optional[bool] = False,
    ):
        """
        Initialises an Experiment object.
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

    def __str__(self) -> str:
        """
        Returns a string representation of the Experiment object.

        Returns:
            str
                A string representation of the Experiment object.
        """
        s = "\n>>> Experiment\n"
        s += f"    * Dataset: {self.dataset.upper()}\n"
        s += f"    * Overwrite processing: {self.overwrite_processing}\n"
        s += f"    * Experiments on: {self.test_split}\n"
        s += f"    * Run end-to-end REL experiments: {self.rel_experiments}"
        return s

    def load_data(self) -> dict:
        """
        Load the data already processed in a previous run of the code, using
        the same parameters.

        Returns:
            dict: A dictionary where the processed data is stored.
        """

        output_path = os.path.join(self.data_path, self.dataset, self.myner.model)

        # Add the candidate experiment info to the path:
        cand_approach = self.myranker.method
        if self.myranker.method == "deezymatch":
            cand_approach += "+" + str(self.myranker.deezy_parameters["num_candidates"])
            cand_approach += "+" + str(
                self.myranker.deezy_parameters["selection_threshold"]
            )

        output_processed_data = dict()
        try:
            with open(output_path + "_ner_predictions.json") as fr:
                output_processed_data["preds"] = json.load(fr)
            with open(output_path + "_gold_standard.json") as fr:
                output_processed_data["trues"] = json.load(fr)
            with open(output_path + "_ner_skyline.json") as fr:
                output_processed_data["skys"] = json.load(fr)
            with open(output_path + "_gold_positions.json") as fr:
                output_processed_data["gold_tok"] = json.load(fr)
            with open(output_path + "_dict_sentences.json") as fr:
                output_processed_data["dSentences"] = json.load(fr)
            with open(output_path + "_dict_metadata.json") as fr:
                output_processed_data["dMetadata"] = json.load(fr)
            with open(output_path + "_pred_mentions.json") as fr:
                output_processed_data["dMentionsPred"] = json.load(fr)
            with open(output_path + "_gold_mentions.json") as fr:
                output_processed_data["dMentionsGold"] = json.load(fr)
            with open(output_path + "_candidates_" + cand_approach + ".json") as fr:
                output_processed_data["dCandidates"] = json.load(fr)

            return output_processed_data
        except FileNotFoundError:
            print("File not found, process data.")
            return dict()

    def prepare_data(self) -> dict:
        """
        Function that prepares the data for the experiments.

        Returns:
            dict
                The processed data dictionary, containing predicted mentions,
                gold standard, REL end-to-end processing, candidates, which
                can be used later for linking.
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
        self.processed_data = self.store_processed_data(
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
        self.store_results(task="ner", how_split="originalsplit")

        return self.processed_data

    def store_processed_data(
        self,
        preds: dict,
        trues: dict,
        skys: dict,
        gold_tok: dict,
        dSentences: dict,
        dMetadata: dict,
        dMentionsPred: dict,
        dMentionsGold: dict,
        dCandidates: dict,
    ) -> dict:
        """
        Stores all the postprocessed data as JSON files and returns a dictionary
        containing all processed data.

        Arguments:
            experiment (experiment.Experiment): An experiment object.
            preds (dict): A dictionary of tokens with predictions per sentence.
            trues (dict): A dictionary of tokens with gold standard annotations per
                sentence.
            skys (dict): A dictionary of tokens representing the skyline per
                sentence.
            gold_tok (dict): A dictionary of tokens with gold standard annotations
                as dictionaries per sentence.
            dSentences (dict): A dictionary mapping a sentence ID to the
                corresponding text.
            dMetadata (dict): A dictionary mapping a sentence ID to associated
                metadata.
            dMentionsPred (dict): A dictionary of predicted mentions per sentence.
            dMentionsGold (dict): A dictionary of gold standard mentions per
                sentence.
            dCandidates (dict): A dictionary of candidates per mention in a
                sentence.

        Returns:
            dict:
                A dictionary containing all processed data (predictions, gold
                standard, skyline, candidates) in one place.

        Note:
            This function also creates one JSON file per dictionary, stored in
            ``outputs/data``.
        """
        data_path = self.data_path
        dataset = self.dataset
        model_name = self.myner.model
        output_path = data_path + dataset + "/" + model_name

        cand_approach = self.myranker.method
        if self.myranker.method == "deezymatch":
            cand_approach += "+" + str(self.myranker.deezy_parameters["num_candidates"])
            cand_approach += "+" + str(
                self.myranker.deezy_parameters["selection_threshold"]
            )

        # Store NER predictions using a specific NER model:
        with open(output_path + "_ner_predictions.json", "w") as fw:
            json.dump(preds, fw)

        # Store gold standard:
        with open(output_path + "_gold_standard.json", "w") as fw:
            json.dump(trues, fw)

        # Store NER skyline:
        with open(output_path + "_ner_skyline.json", "w") as fw:
            json.dump(skys, fw)

        # Store gold tokenisation positions:
        with open(output_path + "_gold_positions.json", "w") as fw:
            json.dump(gold_tok, fw)

        # Store the dictionary of sentences:
        with open(output_path + "_dict_sentences.json", "w") as fw:
            json.dump(dSentences, fw)

        # Store the dictionary of metadata per sentence:
        with open(output_path + "_dict_metadata.json", "w") as fw:
            json.dump(dMetadata, fw)

        # Store the dictionary of predicted results:
        with open(output_path + "_pred_mentions.json", "w") as fw:
            json.dump(dMentionsPred, fw)

        # Store the dictionary of gold standard:
        with open(output_path + "_gold_mentions.json", "w") as fw:
            json.dump(dMentionsGold, fw)

        # Store the dictionary of gold standard:
        with open(output_path + "_candidates_" + cand_approach + ".json", "w") as fw:
            json.dump(dCandidates, fw)

        dict_processed_data = dict()
        dict_processed_data["preds"] = preds
        dict_processed_data["trues"] = trues
        dict_processed_data["skys"] = skys
        dict_processed_data["gold_tok"] = gold_tok
        dict_processed_data["dSentences"] = dSentences
        dict_processed_data["dMetadata"] = dMetadata
        dict_processed_data["dMentionsPred"] = dMentionsPred
        dict_processed_data["dMentionsGold"] = dMentionsGold
        dict_processed_data["dCandidates"] = dCandidates
        return dict_processed_data

    def create_mentions_df(self) -> pd.DataFrame:
        """
        Create a dataframe for the linking experiment, with one mention per row.

        Returns:
            pandas.DataFrame:
                A dataframe with one mention per row, and containing all relevant
                information for subsequent steps (i.e. for linking).

        Note:
            This function also creates a TSV file in the
            ``outputs/data/[dataset]/`` folder.
        """
        dMentions = self.processed_data["dMentionsPred"]
        dGoldSt = self.processed_data["dMentionsGold"]
        dSentences = self.processed_data["dSentences"]
        dMetadata = self.processed_data["dMetadata"]
        dCandidates = self.processed_data["dCandidates"]

        cand_approach = self.myranker.method
        if self.myranker.method == "deezymatch":
            cand_approach += "+" + str(self.myranker.deezy_parameters["num_candidates"])
            cand_approach += "+" + str(
                self.myranker.deezy_parameters["selection_threshold"]
            )

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
                    candidates = dCandidates[sentence_id].get(
                        mention["mention"], dict()
                    )

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
                            candidates,
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
                "candidates",
            ],
            data=rows,
        )

        output_path = (
            self.data_path + self.dataset + "/" + self.myner.model + "_" + cand_approach
        )

        # List of columns to merge (i.e. columns where we have indicated
        # out data splits), and "article_id", the columns on which we
        # will merge the data:
        keep_columns = [
            "article_id",
            "apply",
            "originalsplit",
            "withouttest",
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
        processed_df.to_csv(output_path + "_mentions.tsv", sep="\t")

        return processed_df

    def store_results(
        self,
        task: Literal["ner", "linking"],
        how_split: str,
    ) -> None:
        """
        Function which stores the results of an experiment in the format required
        by the HIPE 2020 evaluation scorer.

        Arguments:
            task (Literal["ner", "linking"]): either "ner" or "linking". Store the
                results for just NER or with links as well.
            how_split (str): which way of splitting the data are we using?
                It could be the ``"originalsplit"`` or ``"Ashton1860"``, for
                example, which would mean that ``"Ashton1860"`` is left out for
                test only.

        Returns:
            None.
        """
        hipe_scorer_results_path = os.path.join(self.results_path, self.dataset)
        Path(hipe_scorer_results_path).mkdir(parents=True, exist_ok=True)

        scenario_name = ""
        if task == "ner":
            scenario_name += task + "_" + self.myner.model + "_"
        if task == "linking":
            scenario_name += task + "_" + self.myner.model + "_"
            cand_approach = self.myranker.method
            if self.myranker.method == "deezymatch":
                cand_approach += "+" + str(
                    self.myranker.deezy_parameters["num_candidates"]
                )
                cand_approach += "+" + str(
                    self.myranker.deezy_parameters["selection_threshold"]
                )

            scenario_name += cand_approach + "_" + how_split + "_"

        link_approach = self.mylinker.method
        if self.mylinker.method == "reldisamb":
            if self.mylinker.rel_params["with_publication"]:
                link_approach += "+wpubl"
            if self.mylinker.rel_params["without_microtoponyms"]:
                link_approach += "+wmtops"
            if self.mylinker.rel_params["do_test"]:
                link_approach += "_test"

        # Find article ids of the corresponding test set (e.g. 'dev' of the original split,
        # 'test' of the Ashton1860 split, etc):
        all = self.dataset_df
        test_articles = list(all[all[how_split] == "test"].article_id.unique())
        test_articles = [str(art) for art in test_articles]

        # Store predictions results formatted for CLEF-HIPE scorer:
        preds_name = link_approach if task == "linking" else "preds"
        process_data.store_for_scorer(
            hipe_scorer_results_path,
            scenario_name + preds_name,
            self.processed_data["preds"],
            test_articles,
        )

        # Store gold standard results formatted for CLEF-HIPE scorer:
        process_data.store_for_scorer(
            hipe_scorer_results_path,
            scenario_name + "trues",
            self.processed_data["trues"],
            test_articles,
        )

        if task == "linking":
            # If task is "linking", store the skyline results (but not for the
            # ranking method of REL):
            process_data.store_for_scorer(
                hipe_scorer_results_path,
                scenario_name + "skys",
                self.processed_data["skys"],
                test_articles,
            )

    def linking_experiments(self) -> None:
        """
        Run entity linking experiments on the processed data.

        This function performs the entity linking experiments using the
        prepared data according to the different configurations of the
        recogniser, ranker and linker. The experiments are performed on
        different data splits and store the results in the specified format
        required by the HIPE scorer. Additionally, it provides an option to
        run end-to-end REL experiments.

        Returns:
            None.

        Note:
            The results of the experiments are stored in the
            ``self.processed_data`` attribute of the Experiment instance.
        """
        # Create a mention-based dataframe for the linking experiments:
        processed_df = self.create_mentions_df()
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
        if self.test_split == "apply":
            list_test_splits = ["apply"]

        # ------------------------------------------
        # Iterate over each linking experiments, each will have its own
        # results file:
        for split in list_test_splits:
            original_df = self.dataset_df
            processed_df = self.processed_data["processed_df"]

            test_original = original_df[original_df[split] == "test"]
            test_processed = processed_df[processed_df[split] == "test"]

            if split == "apply":
                # This is not used in the experiments: in the "apply" mode, we are
                # training on what would be train+dev in the originalsplit, and
                # leave test for development. We're just testing on dev itself
                # to avoid the code failing. The model trained with this scenario
                # should just be used with new data not in the experiments.
                test_original = original_df[original_df[split] == "dev"]
                test_processed = processed_df[processed_df[split] == "dev"]

            # Get ids of articles in each split:
            test_article_ids = list(test_original.article_id.astype(str))

            # Train a linking model if needed (it requires myranker to generate potential
            # candidates to the training set):
            print("Train EL model using:", split)
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
            self.store_results(
                task="linking",
                how_split=split,
            )

        # -----------------------------------------------
        # Run end-to-end REL experiments:
        if self.rel_experiments == True:
            from utils import rel_e2e

            rel_e2e.run_rel_experiments(self)
