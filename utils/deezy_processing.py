import glob
import itertools
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import gensim
import gensim.downloader
from DeezyMatch import combine_vecs
from DeezyMatch import inference as dm_inference
from DeezyMatch import train as dm_train
from gensim.models import Word2Vec
from thefuzz import fuzz
from tqdm import tqdm


def obtain_matches(
    word: str,
    english_words: List[str],
    sims: List[str],
    fuzz_ratio_threshold: Optional[Union[float, int]] = 70,
) -> Tuple[List[str], List[str]]:
    """
    Classifies the top 100 nearest neighbors of the given word into positive
    and negative matches (or discards them).

    Arguments:
        word (str): The input word.
        english_words (list): A list of English words as strings.
        sims (list): The list of 100 nearest neighbors from the OCR word2vec
            model.
        fuzz_ratio_threshold (float): The threshold used for
            `thefuzz.fuzz.ratio <https://github.com/seatgeek/thefuzz#simple-ratio>`.
            If the nearest neighbor word is an existing English word and the
            string similarity is below ``fuzz_ratio_threshold``, it is considered
            a negative match, i.e. not an OCR variation. Defaults to ``70``.

    Returns:
        Tuple[List[str], List[str]]: A tuple that contains two lists:

            #. The first list consists of *positive* matches for the input
               word.
            #. The second list consists of *negative* matches, a list of
               negative matches for the input word.
    """
    negative = []
    positive = [word]
    for nn_word in sims:
        # If one word is not a subset of another:
        if not nn_word in word and not word in nn_word:
            # Split both the word and the nearest neighbour in two parts: the
            # idea is that both parts should be equally similar or equally
            # dissimilar, in order to consider them as positive or negative
            # matches (e.g. "careless" and "listless" are two clear words but
            # have high string similarity due to a big chunk of the word--the
            # suffix--being identical):
            nn_word_1 = nn_word[: len(nn_word) // 2]
            nn_word_2 = nn_word[len(nn_word) // 2 :]
            word_1 = word[: len(word) // 2]
            word_2 = word[len(word) // 2 :]

            # If the nearest neighbour word is a word of the English language
            # and the string similarity is less than 0.50, we consider it a
            # negative match (i.e. not an OCR variation):
            if (
                nn_word in english_words
                and fuzz.ratio(nn_word_1, word_1) < (100 - fuzz_ratio_threshold)
                and fuzz.ratio(nn_word_2, word_2) < (100 - fuzz_ratio_threshold)
                and -2 <= (len(word) - len(nn_word)) <= 2  # Similar length of words
            ):
                negative.append(nn_word)

            # If the nearest neighbour word is not a word of the English
            # language and the string similarity is more than 0.50, we
            # consider it a positive match (i.e. an OCR variation):
            if (
                not nn_word in english_words
                and fuzz.ratio(nn_word_1, word_1) > fuzz_ratio_threshold
                and fuzz.ratio(nn_word_2, word_2) > fuzz_ratio_threshold
            ):
                positive.append(nn_word)

                # Artificially add white space, hyphen or dot (1/50) at a
                # random position:
                spc = [" ", ".", "-"] + [""] * 50
                randomch = random.choice(spc)
                randompos = random.randint(0, len(word) - 1)
                newx = word[:randompos] + randomch + word[randompos:]
                positive.append(newx)

    return positive, negative


def create_training_set(
    deezy_parameters: dict, strvar_parameters: dict, wikidata_to_mentions: dict
) -> None:
    """
    Create a training set for DeezyMatch consisting of positive and negative
    string matches.

    Given a word2vec model trained on OCR data and a list of words in the
    English language, this function creates a training set for DeezyMatch.
    The training set contains pairs of strings, where a positive match is an
    English word and its OCR variation (obtained from the top N word2vec
    neighbours and filtered by string similarity), and a negative
    match is an English word and a randomly selected OCR token.

    Arguments:
        deezy_parameters (dict): Dictionary of DeezyMatch parameters
            for model training.
        strvar_parameters (dict): Dictionary of string variation
            parameters required to create a DeezyMatch training dataset.
        wikidata_to_mentions (dict): Mapping between Wikidata IDs and mentions.

    Returns:
        None.

    Note:
        This function creates a new file with the string pairs dataset called
        ``w2v_ocr_pairs.txt`` inside the folder path defined as ``dm_path`` in
        the DeezyMatch parameters passed in setting up the ranker passed to
        this function as ``myranker``.
    """

    # Path to the output string pairs dataset:
    string_matching_filename = os.path.join(
        deezy_parameters["dm_path"], "data", "w2v_ocr_pairs.txt"
    )
    if deezy_parameters["do_test"] == True:
        string_matching_filename = os.path.join(
            deezy_parameters["dm_path"], "data", "w2v_ocr_pairs_test.txt"
        )

    Path("/".join(string_matching_filename.split("/")[:-1])).mkdir(
        parents=True, exist_ok=True
    )

    # If the dataset exists, do nothing:
    if (
        Path(string_matching_filename).exists()
        and strvar_parameters["overwrite_dataset"] == False
    ):
        print("The string match dataset already exists!")
        return None

    print("Create a string match dataset!")
    print(">>> Loading words in the English language.")
    glove_vectors = gensim.downloader.load("glove-wiki-gigaword-50")

    # Words in the English language:
    english_words = set(glove_vectors.index_to_key)

    print(">>> Creating a dataset of positive and negative matches.")

    # These are w2v models trained on OCR data. In our case, we have
    # w2v models trained on historical newspapers from 1800s, 1830s,
    # and 1860s, by Nilo (data from LwM newspapers).
    positive_matches = []
    negative_matches = []
    for path2model in glob.glob(
        os.path.join(
            strvar_parameters["w2v_ocr_path"],
            strvar_parameters["w2v_ocr_model"],
            "w2v.model",
        )
    ):
        # Read the word2vec model:
        model = Word2Vec.load(path2model)

        # Words in the embeddings:
        w2v_words = list(model.wv.index_to_key)

        # filter w2v_words
        if deezy_parameters["do_test"] == True:
            seedwords_cutoff = 5
            w2v_words = w2v_words[:seedwords_cutoff]

        # For each word in the w2v model, keep likely positive and negative
        # matches:
        for word in tqdm(w2v_words):
            # For each word in the w2v model that is longer than 4 characters
            # and is a word in the English language:
            if (
                len(word) >= strvar_parameters["min_len"]
                and len(word) <= strvar_parameters["max_len"]
                and word in english_words
            ):
                # Get the top 100 nearest neighbors
                sims = model.wv.most_similar(word, topn=100)
                sims = [
                    x[0]
                    for x in sims
                    if strvar_parameters["max_len"]
                    >= len(x[0])
                    >= strvar_parameters["min_len"]
                ]
                # Distinguish between positive and negative matches, where
                # * a positive match is an OCR word variation
                # * a negative match is a different word
                positive, negative = obtain_matches(
                    word,
                    english_words,
                    sims,
                    fuzz_ratio_threshold=strvar_parameters["ocr_threshold"],
                )
                # We should have the same number of positive matches as
                # negative:
                shortest_length = min([len(positive), len(negative)])
                negative = negative[:shortest_length]
                positive = positive[:shortest_length]
                # Prepare for writing into file:
                negative_matches += [
                    word + "\t" + x + "\t" + "FALSE\n" for x in negative
                ]
                positive_matches += [
                    word + "\t" + x + "\t" + "TRUE\n" for x in positive
                ]

    if len(positive_matches) == 0:
        print(
            "Warning: You've got an empty list of positive matches. "
            "Check whether the path to the w2v embeddings is correct."
        )
        return None

    # Get variations from wikidata_to_mentions:
    for wq in wikidata_to_mentions.keys():
        mentions = list(wikidata_to_mentions[wq].keys())
        mentions = [
            "".join(m.splitlines())
            for m in mentions
            if strvar_parameters["min_len"] <= len(m) <= strvar_parameters["max_len"]
        ]
        combs = list(itertools.combinations(mentions, 2))
        combs = [
            x
            for x in combs
            if fuzz.ratio(x[0], x[1]) > strvar_parameters["top_threshold"]
        ]
        positive_matches += [c[0] + "\t" + c[1] + "\t" + "TRUE\n" for c in combs]

    if deezy_parameters["do_test"] == True:
        positive_matches = positive_matches[:100]
        negative_matches = negative_matches[:100]

    with open(string_matching_filename, "w") as fw:
        for nm in negative_matches:
            fw.write(nm)
        for pm in positive_matches:
            fw.write(pm)


def train_deezy_model(deezy_parameters: dict) -> None:
    """
    Train a DeezyMatch model using the provided ``myranker`` parameters and
    input files.

    This function trains a DeezyMatch model based on the specified parameters
    in the myranker object and the required input files. If the
    ``overwrite_training`` parameter is set to True or the model does not
    exist, the function will train a new DeezyMatch model.

    Arguments:
        deezy_parameters (dict): Dictionary of DeezyMatch parameters
            for model training.

    Returns:
        None

    Note:
        This function returns a DeezyMatch model, stored in the location
        specified in the DeezyMatch ``input_dfm.yaml`` file.
    """

    # Read the filepaths:
    input_file_path = os.path.join(
        deezy_parameters["dm_path"], "inputs", "input_dfm.yaml"
    )
    dataset_path = os.path.join(
        deezy_parameters["dm_path"], "data", "w2v_ocr_pairs.txt"
    )
    if deezy_parameters["do_test"] == True:
        dataset_path = os.path.join(
            deezy_parameters["dm_path"], "data", f"w2v_ocr_pairs_test.txt"
        )
    model_name = deezy_parameters["dm_model"]

    # Condition for training:
    # (if overwrite is set to True or the model does not exist, train it)
    if (
        deezy_parameters["overwrite_training"] == True
        or not Path(
            os.path.join(
                deezy_parameters["dm_path"],
                "models",
                model_name,
            )
        ).exists()
    ):
        # Training a DeezyMatch model
        dm_train(
            input_file_path=input_file_path,
            dataset_path=dataset_path,
            model_name=model_name,
        )
    else:
        print("The DeezyMatch model is already trained!")


def generate_candidates(deezy_parameters: dict, mentions_to_wikidata: dict) -> None:
    """
    Obtain Wikidata candidates (Wikipedia mentions to Wikidata entities) and
    generate their corresponding vectors.

    This function retrieves Wikidata candidates based on the mentions stored
    in the ``myranker`` object and generates their corresponding vectors using
    the DeezyMatch model. It writes the candidates to a file and generates
    embeddings with the DeezyMatch model.

    Arguments:
        deezy_parameters (dict): Dictionary of DeezyMatch parameters
            for model training.
        mentions_to_wikidata (dict): Mapping between mentions and Wikidata IDs.

    Returns:
        None.

    Note:
        The function saves the candidates to a file and generates embeddings
        using the DeezyMatch model. The resulting vectors are stored in the
        output directories specified in the DeezyMatch parameters passed to
        the ranker passed to this function in the ``myranker`` keyword
        argument.
    """
    deezymatch_outputs_path = deezy_parameters["dm_path"]
    candidates = deezy_parameters["dm_cands"]
    dm_model = deezy_parameters["dm_model"]

    unique_placenames_array = list(set(list(mentions_to_wikidata.keys())))
    unique_placenames_array = [
        " ".join(x.strip().split("\t")) for x in unique_placenames_array if x
    ]

    if deezy_parameters["do_test"] == True:
        unique_placenames_array = unique_placenames_array[:100]

    with open(
        os.path.join(
            deezymatch_outputs_path,
            "data",
            candidates + ".txt",
        ),
        "w",
    ) as f:
        f.write("\n".join(map(str, unique_placenames_array)))

    # Generate vectors for candidates (specified in dataset_path)
    # using a model stored at pretrained_model_path and pretrained_vocab_path
    if (
        deezy_parameters["overwrite_training"] == True
        or not Path(
            os.path.join(
                deezymatch_outputs_path,
                "combined",
                candidates + "_" + dm_model,
            )
        ).is_dir()
    ):
        start_time = time.time()
        dm_inference(
            input_file_path=os.path.join(
                deezymatch_outputs_path, "models", dm_model, "input_dfm.yaml"
            ),
            dataset_path=os.path.join(
                deezymatch_outputs_path, "data", candidates + ".txt"
            ),
            pretrained_model_path=os.path.join(
                deezymatch_outputs_path, "models", dm_model, dm_model + ".model"
            ),
            pretrained_vocab_path=os.path.join(
                deezymatch_outputs_path, "models", dm_model, dm_model + ".vocab"
            ),
            inference_mode="vect",
            scenario=os.path.join(
                deezymatch_outputs_path,
                "candidate_vectors",
                candidates + "_" + dm_model,
            ),
        )
        elapsed = time.time() - start_time
        print("Generate candidate vectors: %s" % elapsed)

    # Combine vectors stored in the scenario in candidates/ and save them in combined/
    if (
        deezy_parameters["overwrite_training"] == True
        or not Path(
            os.path.join(
                deezymatch_outputs_path, "combined", candidates + "_" + dm_model
            )
        ).is_dir()
    ):
        start_time = time.time()
        combine_vecs(
            rnn_passes=["fwd", "bwd"],
            input_scenario=os.path.join(
                deezymatch_outputs_path,
                "candidate_vectors",
                candidates + "_" + dm_model,
            ),
            output_scenario=os.path.join(
                deezymatch_outputs_path, "combined", candidates + "_" + dm_model
            ),
            print_every=100,
        )
        elapsed = time.time() - start_time
        print("Combine candidate vectors: %s" % elapsed)
