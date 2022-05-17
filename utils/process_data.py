import csv
import glob
import hashlib
import json
import pathlib
import re
import urllib
from ast import literal_eval
from pathlib import Path

import pandas as pd

# ------------------------------
# Wiki2Wiki conversion
# ------------------------------

# ------------------------------
# Load wikipedia2wikidata mapper:
path = "/resources/wikipedia/extractedResources/"
wikipedia2wikidata = dict()
if Path(path + "wikipedia2wikidata.json").exists():
    with open(path + "wikipedia2wikidata.json", "r") as f:
        wikipedia2wikidata = json.load(f)
else:
    print("Warning: wikipedia2wikidata.json does not exist.")


# ------------------------------
# Get wikidata ID from wikipedia URL:
def turn_wikipedia2wikidata(wikipedia_title):
    if not wikipedia_title == "NIL" and not wikipedia_title == "*":
        wikipedia_title = wikipedia_title.split("/wiki/")[-1]
        wikipedia_title = urllib.parse.unquote(wikipedia_title)
        wikipedia_title = wikipedia_title.replace("_", " ")
        wikipedia_title = urllib.parse.quote(wikipedia_title)
        if "/" in wikipedia_title or len(wikipedia_title) > 200:
            wikipedia_title = hashlib.sha224(
                wikipedia_title.encode("utf-8")
            ).hexdigest()
        if not wikipedia_title in wikipedia2wikidata:
            print(
                "Warning: "
                + wikipedia_title
                + " is not in wikipedia2wikidata, the wkdt_qid will be None."
            )
        return wikipedia2wikidata.get(wikipedia_title)
    return None


# ------------------------------
# LwM data
# ------------------------------
# ------------------------------
# This function takes a .tsv (webanno 3.0) file and parses it.
# It returns a dictionary (dTokens) that maps each token with
# its position in the sentence and the associated annotations.
# In particular, the keys in dTokens are tuples of two elements
# (the sentence number in the document, and the character position
# of a token in the document). The values of dTokens are tuples
# of six elements (the actual token, the wikipedia url, the
# toponym class, the sentence number in the document, the character
# position of a token in the document, and the character end
# position of a token in the document).
def process_tsv(filepath):

    with open(filepath) as fr:
        lines = fr.readlines()

    # This regex identifies a token-line in the WebAnno 3.0 format:
    regex_annline = r"^[0-9]+\-[0-9]+\t[0-9]+\-[0-9]+\t.*$"

    # This regex identifies annotations that span multiple tokens:
    regex_multmention = r"^(.*)\[([0-9]+)\]"

    multiple_mention = 0
    prev_multmention = 0
    complete_token = ""  # Here we will build the multitoken if toponym has multiple tokens, otherwise keep the token.
    complete_label = ""  # Label corresponding to complete_token
    complete_wkpd = ""  # Wikidata ID corresponding to complete_token
    dMTokens = (
        dict()
    )  # Dictionary of tokens in which multitoken toponyms are joined into one token
    dTokens = (
        dict()
    )  # Dictionary of tokens in which multitoken toponyms remain as separated tokens (BIO scheme)
    sent_pos = 0  # Sentence position
    tok_pos = 0  # Token position
    tok_start = 0  # Token start char
    tok_end = 0  # Token end char
    mtok_start = 0  # Multitoken start char
    mtok_end = 0  # Multitoken end char
    prev_endchar = 0  # Previous token end char

    # Loop over all lines in the file:
    for line in lines:

        # If the line is a token-line:
        if re.match(regex_annline, line):

            bio_label = "O"

            # If the token-line has no annotations, automatically provide
            # them empty annotations:
            if len(line.strip().split("\t")) == 3:
                sent_tmp, tok_tmp, token = line.strip().split("\t")
                wkpd = "_"
                label = "_"
            # Otherwise, split the token-line to its different layers:
            # * sent_tmp has sentence/token position info
            # * tok_tmp has token characters position info
            # * token is the actual token
            # * wkpd is the wikipedia link annotation
            # * label is the toponym class annotation
            else:
                sent_tmp, tok_tmp, token, wkpd, label = line.strip().split("\t")

            # If the annotation corresponds to a multi-token annotation (i.e. WikipediaID string
            # ends with a number enclosed in square brackets, as in "San[1]" and "Francisco[1]"):

            if re.match(regex_multmention, wkpd):

                # This code basically collates multi-token mentions in annotations
                # together. "complete_token" is the resulting multi-token mention,
                # "sent_pos" is the sentence position in the file, "tok_pos" is the
                # token position in the sentence, "mtok_start" is the multi-token

                # character start position in the document, and "tok_end" is the
                # multi-token character end position in the document.
                multiple_mention = int(re.match(regex_multmention, wkpd).group(2))
                complete_label = re.match(regex_multmention, label).group(1)
                complete_wkpd = re.match(regex_multmention, wkpd).group(1)

                # If we identify that we're dealing with a multi-token mention:
                if multiple_mention == prev_multmention:
                    # Preappend as many white spaces as the distance between the end of the
                    # previous token and the start of the current token:
                    complete_token += " " * (
                        int(tok_tmp.split("-")[0]) - int(prev_endchar)
                    )
                    # Append the current token to the complete token:
                    complete_token += token
                    # The complete_token end character will be considered to be the end of
                    # the latest token in the multi-token:
                    tok_start, tok_end = tok_tmp.split("-")
                    mtok_end = tok_tmp.split("-")[1]
                    # Here we keep the end position of the previous token:
                    prev_endchar = int(tok_tmp.split("-")[1])
                    bio_label = "I-" + label
                else:
                    sent_pos, tok_pos = sent_tmp.split("-")
                    tok_start, tok_end = tok_tmp.split("-")
                    mtok_start, mtok_end = tok_tmp.split("-")
                    prev_endchar = int(tok_end)
                    complete_token = token
                    bio_label = "B-" + label
                prev_multmention = multiple_mention

            # If the annotation does not correspond to a multi-token annotation,
            # just keep the token-specific information:
            else:
                sent_pos, tok_pos = sent_tmp.split("-")
                tok_start, tok_end = tok_tmp.split("-")
                mtok_start, mtok_end = tok_tmp.split("-")
                complete_token = token
                complete_label = label
                complete_wkpd = wkpd
                if label and label != "_" and label != "*":
                    bio_label = "B-" + label

            sent_pos = int(sent_pos)
            tok_pos = int(tok_pos)
            tok_start = int(tok_start)
            mtok_start = int(mtok_start)
            tok_end = int(tok_end)
            mtok_end = int(mtok_end)

            bio_label = bio_label.split("[")[0]
            wkpd = wkpd.split("[")[0]

            dMTokens[(sent_pos, mtok_start)] = (
                complete_token,
                complete_wkpd,
                complete_label,
                sent_pos,
                mtok_start,
            )

            dTokens[(sent_pos, tok_start)] = (
                token,
                wkpd,
                bio_label,
                sent_pos,
                tok_start,
                tok_end,
            )

    return dMTokens, dTokens


# ------------------------------
# Given the dictionary of tokens (with their positional information
# in the document and associated annotations), reconstruct all sentences
# in the document (taking into account white spaces to make sure character
# positions match).
def reconstruct_sentences(dTokens):

    complete_sentence = ""  # In this variable we keep each complete sentence
    sentence_id = 0  # In this variable we keep the sentence id
    start_ids = [
        k[1] for k in dTokens
    ]  # Ordered list of token character start positions.
    dSentences = (
        dict()
    )  # In this variable we'll map the sentence id with the complete sentence
    prev_sentid = 0  # Keeping track of previous sentence id (to know when a new sentence is starting)
    dSentenceCharstart = (
        dict()
    )  # In this variable we will map the sentence id with the start character
    # position of the sentence in question

    dIndices = (
        dict()
    )  # In this dictionary we map the token start character in the document with
    # the full mention, the sentence id, and the end character of the mention.

    # In this for-loop, we populate dSentenceCharstart and dIndices:
    for k in dTokens:
        sentence_id = k[0]
        tok_startchar = k[1]
        tok_endchar = dTokens[k][5]
        mention = dTokens[k][0]
        dIndices[tok_startchar] = [mention, sentence_id, tok_endchar]
        if not sentence_id in dSentenceCharstart:
            dSentenceCharstart[sentence_id] = tok_startchar

    # In this for loop, we reconstruct the sentences, tanking into account
    # the different positional informations (and adding white spaces when
    # required):
    for i in range(start_ids[0], len(start_ids) + 1):

        if i < len(start_ids) - 1:

            mention = dIndices[start_ids[i]][0]
            sentence_id = dIndices[start_ids[i]][1]
            mention_endchar = dIndices[start_ids[i]][2]

            if sentence_id != prev_sentid:
                complete_sentence = ""

            if mention_endchar < start_ids[i + 1]:
                complete_sentence += mention + (
                    " " * (start_ids[i + 1] - mention_endchar)
                )
            elif mention_endchar == start_ids[i + 1]:
                complete_sentence += mention

            i = start_ids[i]

        elif i == len(start_ids) - 1:
            if sentence_id != prev_sentid:
                complete_sentence = dIndices[start_ids[-1]][0]
            else:
                complete_sentence += dIndices[start_ids[-1]][0]

        # dSentences is a dictionary where the key is the sentence id (i.e. position)
        # in the document, and the value is a two-element list: the first element is
        # the complete reconstructed sentence, and the second element is the character
        # start position of the sentence in the document:
        dSentences[sentence_id] = [complete_sentence, dSentenceCharstart[sentence_id]]

        prev_sentid = sentence_id

    return dSentences


# ------------------------------
# Process LwM data for training a NER model, where each sentence has an id,
# and a list of tokens and assigned ner_tags using the BIO scheme, e.g.:
# > id: 10813493_1 # document_id + "_" + sentence_id
# > ner_tags: ['B-LOC', 'O']
# > tokens: ['INDIA', '.']
def process_lwm_for_ner(tsv_topres_path):
    lwm_data = []

    for fid in glob.glob(tsv_topres_path + "annotated_tsv/*"):

        filename = fid.split("/")[-1]  # Full document name
        file_id = filename.split("_")[0]  # Document id

        # Dictionary that maps each token in a document with its
        # positional information and associated annotations:
        dMTokens, dTokens = process_tsv(tsv_topres_path + "annotated_tsv/" + filename)

        ner_tags = []
        tokens = []
        prev_sent = 0
        for t in dTokens:
            curr_sent = t[0]
            if curr_sent == prev_sent:
                ner_tags.append(dTokens[t][2])
                tokens.append(dTokens[t][0])
            else:
                if tokens and ner_tags:
                    lwm_data.append(
                        {
                            "id": file_id + "_" + str(prev_sent),
                            "ner_tags": ner_tags,
                            "tokens": tokens,
                        }
                    )
                ner_tags = [dTokens[t][2]]
                tokens = [dTokens[t][0]]
            prev_sent = curr_sent

        # Now append the tokens and ner_tags of the last sentence:
        lwm_data.append(
            {
                "id": file_id + "_" + str(prev_sent),
                "ner_tags": ner_tags,
                "tokens": tokens,
            }
        )

    lwm_df = pd.DataFrame(lwm_data)

    return lwm_df


# ------------------------------
# Process data for performing entity linking, resulting in a dataframe with
# one toponym per row and its annotation and resolution in columns.
def process_lwm_for_linking(tsv_topres_path):

    # Create the dataframe where we will store our annotated
    # data in a format that works better for us:
    df = pd.DataFrame(
        columns=[
            "article_id",
            "sentences",
            "annotations",
            "place",
            "decade",
            "year",
            "ocr_quality_mean",
            "ocr_quality_sd",
            "publication_title",
            "publication_code",
        ]
    )

    metadata_df = pd.read_csv(
        tsv_topres_path + "metadata.tsv", sep="\t", index_col="fname"
    )

    for fid in glob.glob(tsv_topres_path + "annotated_tsv/*"):
        filename = fid.split("/")[-1].split(".tsv")[0]  # Full document name

        # Fields to fill:
        article_id = filename.split("_")[0]
        sentences = []
        annotations = []
        place_publication = metadata_df.loc[filename]["place_publication"]
        decade = int(metadata_df.loc[filename]["issue_date"][:3] + "0")
        year = int(metadata_df.loc[filename]["issue_date"][:4])
        ocr_quality_mean = float(metadata_df.loc[filename]["ocr_quality_mean"])
        ocr_quality_sd = float(metadata_df.loc[filename]["ocr_quality_sd"])
        publication_title = metadata_df.loc[filename]["publication_title"]
        publication_code = str(metadata_df.loc[filename]["publication_code"]).zfill(7)

        # Dictionary that maps each token in a document with its
        # positional information and associated annotations:
        dMTokens, dTokens = process_tsv(
            tsv_topres_path + "annotated_tsv/" + filename + ".tsv"
        )

        # Dictionary of reconstructed sentences:
        dSentences = reconstruct_sentences(dTokens)

        # List of annotation dictionaries:
        mention_counter = 0
        for k in dMTokens:
            # For each annotated token:
            mention, wkpd, label, sent_pos, tok_start = dMTokens[k]
            if not wkpd == "_" and not label == "_":
                current_sentence, sentence_start = dSentences[sent_pos]
                mention_start = tok_start - sentence_start
                # marked_sentence = current_sentence
                # # We try to match the mention to the position in the sentence, and mask the mention
                # # based on the positional information:
                # if marked_sentence[mention_start:mention_start + len(mention)] == mention:
                #     marked_sentence = marked_sentence[:mention_start] + " [MASK] " + marked_sentence[mention_start + len(mention):]
                # # But there's one document that has some weird indices, and one sentence in particular
                # # in which it is not possible to match the mention with the positional information we
                # # got. In this case, we mask based on string matching:
                # else:
                #     marked_sentence = marked_sentence.replace(mention, " [MASK] ")
                # marked_sentence = re.sub(' +', ' ', marked_sentence)

                # Clean Wikidata URL:
                wkpd = wkpd.replace("\\", "")

                # Get Wikidata ID:
                wkdt = turn_wikipedia2wikidata(wkpd)

                # In mentions attached to next token through a dash,
                # keep only the true mention (this has to do with
                # the annotation format)
                if "—" in mention:
                    mention = mention.split("—")[0]

                annotations.append(
                    {
                        "mention_pos": mention_counter,
                        "mention": mention,
                        "entity_type": label,
                        "wkpd_url": wkpd,
                        "wkdt_qid": wkdt,
                        "mention_start": mention_start,
                        "mention_end": mention_start + len(mention),
                        "sent_pos": sent_pos,
                    }
                )

                mention_counter += 1

        for s_index in dSentences:
            # We keep only the sentence, we don't need the character offset of the sentence
            # going forward:
            sentences.append(
                {"sentence_pos": s_index, "sentence_text": dSentences[s_index][0]}
            )

        df_columns_row = [
            article_id,
            sentences,
            annotations,
            place_publication,
            decade,
            year,
            ocr_quality_mean,
            ocr_quality_sd,
            publication_title,
            publication_code,
        ]

        # Convert the row into a pd.Series:
        row = pd.Series(df_columns_row, index=df.columns)

        # And append it to the main dataframe:
        df = pd.concat([df, row.to_frame().T], ignore_index=True)

    return df


def crate_training_for_el(df):

    # Create dataframe by mention:
    rows = []
    for i, row in df.iterrows():
        article_id = row["article_id"]
        place = row["place"]
        year = row["year"]
        ocr_quality_mean = row["ocr_quality_mean"]
        ocr_quality_sd = row["ocr_quality_sd"]
        publication_title = row["publication_title"]
        publication_code = str(row["publication_code"]).zfill(7)
        sentences = literal_eval(row["sentences"])
        annotations = literal_eval(row["annotations"])
        for s in sentences:
            for a in annotations:
                if s["sentence_pos"] == a["sent_pos"]:
                    rows.append(
                        (
                            article_id,
                            s["sentence_pos"],
                            s["sentence_text"],
                            a["mention_pos"],
                            a["mention"],
                            a["wkdt_qid"],
                            a["mention_start"],
                            a["mention_end"],
                            a["entity_type"],
                            year,
                            place,
                            ocr_quality_mean,
                            ocr_quality_sd,
                            publication_title,
                            publication_code,
                        )
                    )

    training_df = pd.DataFrame(
        columns=[
            "article_id",
            "sentence_pos",
            "sentence",
            "mention_pos",
            "mention",
            "wkdt_qid",
            "mention_start",
            "mention_end",
            "entity_type",
            "year",
            "place",
            "ocr_quality_mean",
            "ocr_quality_sd",
            "publication_title",
            "publication_code",
        ],
        data=rows,
    )

    return training_df


# ------------------------------
# HIPE data
# ------------------------------

# Aggregate split entities:
def aggregate_hipe_entities(entity, lEntities):
    newEntity = entity
    # We remove the word index because we're altering it (by joining suffixes)
    newEntity.pop("index", None)
    # If token is part of a multitoken that has already started in the previous token (i.e. I-),
    # then join with previous detected entity, unless a sentence starts with an I- entity (due
    # to incorrect sentence splitting in the original data).
    if lEntities and entity["ne_type"].startswith("I-"):
        prevEntity = lEntities.pop()
        newEntity = {
            "ne_type": prevEntity["ne_type"],
            "word": prevEntity["word"]
            + ((entity["start"] - prevEntity["end"]) * " ")
            + entity["word"],
            "wkdt_qid": entity["wkdt_qid"],
            "start": prevEntity["start"],
            "end": entity["end"],
        }

    lEntities.append(newEntity)
    return lEntities


# ------------------------------
# Process HIPE data for linking:
def process_hipe_for_linking(hipe_path):

    # Create the dataframe where we will store our annotated
    # data in a format that works better for us:
    df = pd.DataFrame(
        columns=[
            "article_id",
            "sentences",
            "annotations",
            "place",
            "decade",
            "year",
            "ocr_quality_mean",
            "ocr_quality_sd",
            "publication_title",
            "publication_code",
        ]
    )

    article_id = ""
    new_sentence = ""
    new_document = []
    dSentences = dict()
    dAnnotations = dict()
    dMetadata = dict()
    char_index = 0
    sent_index = 0
    newspaper_id = ""
    date = ""
    end_sentence = False
    start_document = False
    with open(hipe_path) as fr:
        lines = fr.readlines()
        for line in lines[1:]:
            if line.startswith("# newspaper"):
                newspaper_id = line.split("= ")[-1].strip()
            elif line.startswith("# date"):
                date = int(line.split("= ")[-1].strip()[:4])
            elif line.startswith("# document_id"):
                if new_sentence:
                    new_document.append(new_sentence)
                    sent_index = 0
                new_document = []
                new_sentence = ""
                article_id = line.split("= ")[-1].strip()
                start_document = True
                dMetadata[article_id] = {"newspaper_id": newspaper_id, "date": date}
            elif not line.startswith("#"):
                line = line.strip().split()
                if len(line) == 10:
                    token = line[0]
                    start_char = char_index
                    end_char = start_char + len(token)
                    etag = line[1]
                    elink = line[7]
                    comment = line[-1]

                    if "PySBDSegment" in comment:
                        if end_sentence == True:
                            if start_document == False:
                                new_document.append(new_sentence)
                                sent_index += 1
                            new_sentence = token
                        else:
                            new_sentence += token
                        char_index = 0
                        end_sentence = True

                    elif "NoSpaceAfter" in comment:
                        if end_sentence == True:
                            if start_document == False:
                                new_document.append(new_sentence)
                                sent_index += 1
                            new_sentence = token
                        else:
                            new_sentence += token
                        char_index = end_char
                        end_sentence = False

                    else:
                        if end_sentence == True:
                            if start_document == False:
                                new_document.append(new_sentence)
                                sent_index += 1
                            new_sentence = token
                        else:
                            new_sentence += token
                        new_sentence += " "
                        char_index = end_char + 1
                        end_sentence = False

                    start_document = False

                    if article_id in dAnnotations:
                        if sent_index in dAnnotations[article_id]:
                            dAnnotations[article_id][sent_index].append(
                                (token, etag, elink, start_char, end_char)
                            )
                        else:
                            dAnnotations[article_id][sent_index] = [
                                (token, etag, elink, start_char, end_char)
                            ]
                    else:
                        dAnnotations[article_id] = {
                            sent_index: [(token, etag, elink, start_char, end_char)]
                        }

            if article_id and new_document:
                dSentences[article_id] = new_document

    for k in dSentences:
        sentence_counter = 0
        mention_counter = 0
        dSentencesFile = []
        dAnnotationsFile = []
        for i in range(len(dSentences[k])):
            sentence_counter += 1
            # Populate the dictionary of sentences per file:
            dSentencesFile.append(
                {"sentence_pos": sentence_counter, "sentence_text": dSentences[k][i]}
            )
            # Create a dictionary of multitoken entities for linking:
            annotations = dAnnotations[k][i]
            dAnnotationsTmp = []
            mentions = []
            lAnnotations = []
            predictions = []
            start_sentence_pos = annotations[0][3]
            for a in annotations:
                dAnnotationsTmp.append(
                    {
                        "ne_type": a[1],
                        "word": a[0],
                        "wkdt_qid": a[2],
                        "start": a[3] - start_sentence_pos,
                        "end": a[4] - start_sentence_pos,
                    }
                )

            for a in dAnnotationsTmp:
                predictions = aggregate_hipe_entities(a, lAnnotations)

            for p in predictions:
                if p["ne_type"].lower().endswith("loc"):
                    mentions = {
                        "mention_pos": mention_counter,
                        "mention": p["word"],
                        "entity_type": p["ne_type"].split("-")[-1],
                        "wkdt_qid": p["wkdt_qid"],
                        "mention_start": p["start"],
                        "mention_end": p["end"],
                        "sent_pos": sentence_counter,
                    }

                    dAnnotationsFile.append(mentions)
                    mention_counter += 1

        df_columns_row = [
            k,  # article_id
            dSentencesFile,  # sentences
            dAnnotationsFile,  # annotations
            "",  # place_publication
            int(str(dMetadata[k]["date"])[:3] + "0"),  # decade
            dMetadata[k]["date"],  # year
            None,  # ocr_quality_mean
            None,  # ocr_quality_sd
            "",  # publication_title
            dMetadata[k]["newspaper_id"],  # publication_code
        ]

        # Convert the row into a pd.Series:
        row = pd.Series(df_columns_row, index=df.columns)

        # And append it to the main dataframe:
        df = pd.concat([df, row.to_frame().T], ignore_index=True)

    return df


# ------------------------------
# EVALUATION WITH CLEF SCORER
# ------------------------------

# Skyline
def store_resolution_skyline(dataset, approach, value):
    pathlib.Path("outputs/results/" + dataset + "/").mkdir(parents=True, exist_ok=True)
    skyline = open("outputs/results/" + dataset + "/" + approach + ".skyline", "w")
    skyline.write(str(value))
    skyline.close()


# Storing results for evaluation using the CLEF-HIPE scorer
def store_results_hipe(dataset, dataresults, dresults):
    """
    Store results in the right format to be used by the CLEF-HIPE
    scorer: https://github.com/impresso/CLEF-HIPE-2020-scorer.

    Assuming the CLEF-HIPE scorer is stored in ../CLEF-HIPE-2020-scorer/,
    run scorer as follows:
    For NER:
    > python ../CLEF-HIPE-2020-scorer/clef_evaluation.py --ref outputs/results/lwm-true_bundle2_en_1.tsv --pred outputs/results/lwm-pred_bundle2_en_1.tsv --task nerc_coarse --outdir outputs/results/
    For EL:
    > python ../CLEF-HIPE-2020-scorer/clef_evaluation.py --ref outputs/results/lwm-true_bundle2_en_1.tsv --pred outputs/results/lwm-pred_bundle2_en_1.tsv --task nel --outdir outputs/results/
    """
    pathlib.Path("outputs/results/" + dataset + "/").mkdir(parents=True, exist_ok=True)
    # Bundle 2 associated tasks: NERC-coarse and NEL
    with open(
        "outputs/results/" + dataset + "/" + dataresults + "_bundle2_en_1.tsv", "w"
    ) as fw:
        fw.write(
            "TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tNE-FINE-LIT\tNE-FINE-METO\tNE-FINE-COMP\tNE-NESTED\tNEL-LIT\tNEL-METO\tMISC\n"
        )
        for sent_id in dresults:
            fw.write("# sentence_id = " + sent_id + "\n")
            for t in dresults[sent_id]:
                elink = t[2]
                if t[2].startswith("B-"):
                    elink = t[2].replace("B-", "")
                elif t[2].startswith("I-"):
                    elink = t[2].replace("I-", "")
                elif t[1] != "O":
                    elink = "NIL"
                fw.write(
                    t[0]
                    + "\t"
                    + t[1]
                    + "\t"
                    + t[1]
                    + "\tO\tO\tO\tO\t"
                    + elink
                    + "\t"
                    + elink
                    + "\tO\n"
                )
            fw.write("\n")


def read_gold_standard(path):
    if Path(path).is_file():
        with open(path) as f:
            d = json.load(f)
            return d
    else:
        print(
            "The tokenised gold standard is missing. You should first run the LwM baselines."
        )
        exit()


def match_wikipedia_to_wikidata(wiki_title):
    el = urllib.parse.quote(wiki_title.replace("_", " "))
    try:
        el = wikipedia2wikidata[el]
        return el
    except Exception:
        # to be checked but it seems some Wikipedia pages are not in our Wikidata
        # see for instance Zante%2C%20California
        return "O"


# check NER labels in REL
accepted_labels = {"LOC"}


def match_ent(pred_ents, start, end, prev_ann):
    for ent in pred_ents:
        if ent[-1] in accepted_labels:
            st_ent = ent[0]
            len_ent = ent[1]
            if start >= st_ent and end <= (st_ent + len_ent):
                if prev_ann == ent[-1]:
                    ent_pos = "I-"
                else:
                    ent_pos = "B-"
                    prev_ann = ent[-1]

                n = ent_pos + ent[-1]
                try:
                    el = ent_pos + match_wikipedia_to_wikidata(ent[3])
                except Exception as e:
                    print(e)
                    # to be checked but it seems some Wikipedia pages are not in our Wikidata
                    # see for instance Zante%2C%20California
                    return n, "O", ""
                return n, el, prev_ann
    return "O", "O", ""


# Split list into equal parts:
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))