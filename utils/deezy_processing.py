import glob
import itertools
import os
import sys
import random
import time
from pathlib import Path

import gensim
import gensim.downloader
from gensim.models import Word2Vec

# Resources for English words (these will need to change if language is different):
from DeezyMatch import combine_vecs
from DeezyMatch import inference as dm_inference
from DeezyMatch import train as dm_train

from thefuzz import fuzz
from tqdm import tqdm

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import get_data


def obtain_matches(word, english_words, sims, fuzz_ratio_threshold=70):
    """
    Given a word and the top 100 nearest neighbours, separate into positive and negative matches.

    Arguments:
        word (str): a word.
        sims (list): the list of 100 nearest neighbours from the OCR word2vec model.
        fuzz_ratio_threshold (float): threshold for fuzz.ratio
            If the nearest neighbour word is a word of the English language
            and the string similarity is less than fuzz_ratio_threshold, we consider it a
            negative match (i.e. not an OCR variation)

    Returns:
        positive (list): a list of postive matches.
        negative (list): a list of negative matches.
    """
    negative = []
    positive = [word]
    for nn_word in sims:
        # If one word is not a subset of another:
        if not nn_word in word and not word in nn_word:
            # Split both the word and the nearest neighbour in two parts: the idea is that both
            # parts should be equally similar or equally dissimilar, in order to consider them
            # as positive or negative matches (e.g. "careless" and "listless" are two clear words
            # but have high string similarity due to a big chunk of the word---the suffix---being
            # identical):
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
            ):
                negative.append(nn_word)
            # If the nearest neighbour word is not a word of the English language
            # and the string similarity is more than 0.50, we consider it a
            # positive match (i.e. an OCR variation):
            if (
                not nn_word in english_words
                and fuzz.ratio(nn_word_1, word_1) > fuzz_ratio_threshold
                and fuzz.ratio(nn_word_2, word_2) > fuzz_ratio_threshold
            ):
                positive.append(nn_word)

                # Artificially add white space, hyphen or dot (1/50) at a random position:
                spc = [" ", ".", "-"] + [""] * 50
                randomch = random.choice(spc)
                randompos = random.randint(0, len(word) - 1)
                newx = word[:randompos] + randomch + word[randompos:]
                positive.append(newx)

    return positive, negative


def create_training_set(myranker):
    """
    Given a word2vec model trained on OCRd data, and given a list of words
    in the English language, this function creates a training set for
    DeezyMatch, consisting of positive and negative matches (where a
    positive match is the English word and its OCR variation, taken
    from the top N w2v neighbours and fitered by string similarity;
    and a negative match is an English word and a random OCR token.

    Arguments:
        myranker: a Ranker object.

    This function creates a new file with the string pairs dataset.
    """

    # Path to the output string pairs dataset:
    string_matching_filename = os.path.join(
        myranker.deezy_parameters["dm_path"], "data", f"w2v_ocr_pairs.txt"
    )
    if myranker.deezy_parameters["do_test"] == True:
        string_matching_filename = os.path.join(
            myranker.deezy_parameters["dm_path"], "data", f"w2v_ocr_pairs_test.txt"
        )

    dm_model_path = os.path.join(
        myranker.deezy_parameters["dm_path"],
        "models",
        myranker.deezy_parameters["dm_model"],
    )

    Path("/".join(string_matching_filename.split("/")[:-1])).mkdir(
        parents=True, exist_ok=True
    )

    # If the dataset exists, do nothing:
    if (
        Path(string_matching_filename).exists()
        and myranker.strvar_parameters["overwrite_dataset"] == False
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
            myranker.strvar_parameters["w2v_ocr_path"],
            myranker.strvar_parameters["w2v_ocr_model"],
            "w2v.model",
        )
    ):

        # Read the word2vec model:
        model = Word2Vec.load(path2model)

        # Words in the embeddings:
        w2v_words = list(model.wv.index_to_key)

        # filter w2v_words
        if myranker.deezy_parameters["do_test"] == True:
            seedwords_cutoff = 5
            w2v_words = w2v_words[:seedwords_cutoff]

        # For each word in the w2v model, keep likely positive and negative matches:
        for word in tqdm(w2v_words):
            # For each word in the w2v model that is longer than 4 characters and
            # is a word in the English language:
            if (
                len(word) >= myranker.strvar_parameters["min_len"]
                and len(word) <= myranker.strvar_parameters["max_len"]
                and word in english_words
            ):
                # Get the top 100 nearest neighbors
                sims = model.wv.most_similar(word, topn=100)
                sims = [
                    x[0]
                    for x in sims
                    if myranker.strvar_parameters["max_len"]
                    >= len(x[0])
                    >= myranker.strvar_parameters["min_len"]
                ]
                # Distinguist between positive and negative matches, where
                # * a positive match is an OCR word variation
                # * a negative match is a different word
                positive, negative = obtain_matches(
                    word,
                    english_words,
                    sims,
                    fuzz_ratio_threshold=myranker.strvar_parameters["ocr_threshold"],
                )
                # We should have the same number of positive matches as negative:
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

    # Get variations from mentions_to_wikidata:
    for wq in myranker.wikidata_to_mentions.keys():
        mentions = list(myranker.wikidata_to_mentions[wq].keys())
        mentions = [
            "".join(m.splitlines())
            for m in mentions
            if myranker.strvar_parameters["min_len"]
            <= len(m)
            <= myranker.strvar_parameters["max_len"]
        ]
        combs = list(itertools.combinations(mentions, 2))
        combs = [
            x
            for x in combs
            if fuzz.ratio(x[0], x[1]) > myranker.strvar_parameters["top_threshold"]
        ]
        positive_matches += [c[0] + "\t" + c[1] + "\t" + "TRUE\n" for c in combs]

    if myranker.deezy_parameters["do_test"] == True:
        positive_matches = positive_matches[:100]
        negative_matches = negative_matches[:100]

    with open(string_matching_filename, "w") as fw:
        for nm in negative_matches:
            fw.write(nm)
        for pm in positive_matches:
            fw.write(pm)


def train_deezy_model(myranker):
    """
    This function trains a DeezyMatch model given the parameters of a myranker
    object and the required input files.

    Arguments:
        myranker: a Ranker object.

    This function returns a DeezyMatch model, stored in the location specified
    in the DeezyMatch input_dfm.yaml file.
    """

    # Read the filepaths:
    input_file_path = os.path.join(
        myranker.deezy_parameters["dm_path"], "inputs", "input_dfm.yaml"
    )
    dataset_path = os.path.join(
        myranker.deezy_parameters["dm_path"], "data", "w2v_ocr_pairs.txt"
    )
    if myranker.deezy_parameters["do_test"] == True:
        dataset_path = os.path.join(
            myranker.deezy_parameters["dm_path"], "data", f"w2v_ocr_pairs_test.txt"
        )
    model_name = myranker.deezy_parameters["dm_model"]

    # Condition for training:
    # (if overwrite is set to True or the model does not exist, train it)
    if (
        myranker.deezy_parameters["overwrite_training"] == True
        or not Path(
            os.path.join(
                myranker.deezy_parameters["dm_path"],
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


def generate_candidates(myranker):
    """
    Obtain Wikidata candidates (wikipedia mentions to wikidata
    entities) and generate their corresponding vectors.

    Arguments:
        myranker: a Ranker object.

    This function returns a file with all candidates, and the embeddings
    generated with the DeezyMatch model.
    """
    deezymatch_outputs_path = myranker.deezy_parameters["dm_path"]
    candidates = myranker.deezy_parameters["dm_cands"]
    dm_model = myranker.deezy_parameters["dm_model"]

    unique_placenames_array = list(set(list(myranker.mentions_to_wikidata.keys())))
    unique_placenames_array = [
        " ".join(x.strip().split("\t")) for x in unique_placenames_array if x
    ]

    if myranker.deezy_parameters["do_test"] == True:
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
    if not Path(
        os.path.join(
            deezymatch_outputs_path,
            "combined",
            candidates + "_" + dm_model,
        )
    ).is_dir():
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
    if not Path(
        os.path.join(deezymatch_outputs_path, "combined", candidates + "_" + dm_model)
    ).is_dir():
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
