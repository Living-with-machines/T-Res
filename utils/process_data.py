import csv
import glob
import hashlib
import json
import pathlib
import re
import urllib
from pathlib import Path

import pandas as pd

# Load wikipedia2wikidata mapper:
path = "/resources/wikipedia/extractedResources/"
wikipedia2wikidata = dict()
if Path(path + "wikipedia2wikidata.json").exists():
    with open(path + "wikipedia2wikidata.json", "r") as f:
        wikipedia2wikidata = json.load(f)
else:
    print("Warning: wikipedia2wikidata.json does not exist.")


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
                    complete_token += " " * (int(tok_tmp.split("-")[0]) - int(prev_endchar))
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
                mtok_end,
            )
            dTokens[(sent_pos, tok_start)] = (token, wkpd, bio_label, sent_pos, tok_start, tok_end)

    return dMTokens, dTokens


# ------------------------------
# Given the dictionary of tokens (with their positional information
# in the document and associated annotations), reconstruct all sentences
# in the document (taking into account white spaces to make sure character
# positions match).
def reconstruct_sentences(dTokens):

    complete_sentence = ""  # In this variable we keep each complete sentence
    sentence_id = 0  # In this variable we keep the sentence id
    start_ids = [k[1] for k in dTokens]  # Ordered list of token character start positions.
    dSentences = dict()  # In this variable we'll map the sentence id with the complete sentence
    prev_sentid = (
        0  # Keeping track of previous sentence id (to know when a new sentence is starting)
    )
    dSentenceCharstart = (
        dict()
    )  # In this variable we will map the sentence id with the start character
    # position of the sentence in question

    dIndices = dict()  # In this dictionary we map the token start character in the document with
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
                complete_sentence += mention + (" " * (start_ids[i + 1] - mention_endchar))
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
# Get wikidata ID from wikipedia URL:
def turn_wikipedia2wikidata(wikipedia_title):
    if not wikipedia_title == "*":
        wikipedia_title = wikipedia_title.split("/wiki/")[-1]
        wikipedia_title = urllib.parse.unquote(wikipedia_title)
        wikipedia_title = wikipedia_title.replace("_", " ")
        wikipedia_title = urllib.parse.quote(wikipedia_title)
        if "/" in wikipedia_title or len(wikipedia_title) > 200:
            wikipedia_title = hashlib.sha224(wikipedia_title.encode("utf-8")).hexdigest()
        return wikipedia2wikidata.get(wikipedia_title)
    return None


# ------------------------------

# Populate dataframe rows:
def create_lwmdf_row(
    mention_values, file_id, publ_place, publ_decade, mention_counter, dSentences
):

    mention, wkpd, label, sent_pos, tok_start, tok_end = mention_values
    if not wkpd == "_" and not label == "_":
        mention_counter += 1
        current_sentence, sentence_start = dSentences[sent_pos]
        mention_start = tok_start - sentence_start
        marked_sentence = current_sentence
        # We try to match the mention to the position in the sentence, and mask the mention
        # based on the positional information:
        if marked_sentence[mention_start : mention_start + len(mention)] == mention:
            marked_sentence = (
                marked_sentence[:mention_start]
                + " [MASK] "
                + marked_sentence[mention_start + len(mention) :]
            )
        # But there's one document that has some weird indices, and one sentence in particular
        # in which it is not possible to match the mention with the positional information we
        # got. In this case, we mask based on string matching:
        else:
            marked_sentence = marked_sentence.replace(mention, " [MASK] ")
        marked_sentence = re.sub(" +", " ", marked_sentence)
        prev_sentence = ""
        if sent_pos - 1 in dSentences:
            prev_sentence = dSentences[sent_pos - 1][0]
        next_sentence = ""
        if sent_pos + 1 in dSentences:
            next_sentence = dSentences[sent_pos + 1][0]

        wkpd = wkpd.replace("\\", "")
        wkdt = turn_wikipedia2wikidata(wkpd)

        # In mentions attached to next token through a dash,
        # keep only the true mention (this has to do with
        # the annotation format)
        if "—" in mention:
            mention = mention.split("—")[0]

        row = [
            mention_counter,
            sent_pos,
            file_id,
            publ_place,
            publ_decade,
            prev_sentence,
            current_sentence,
            marked_sentence,
            next_sentence,
            mention,
            label,
            wkpd,
            wkdt,
            mention_start,
            mention_start + len(mention),
        ]

        return row

    return False


# ------------------------------
# Add columns to the LwM dataframe containing other mentions of toponyms in the sentence or document:
def add_cotoponyms(df, article_id, sent_id):
    sentence_toponyms = list(df[df["article_id"] == article_id].mention)
    document_toponyms = list(
        df[(df["article_id"] == article_id) & (df["sent_id"] == sent_id)].mention
    )
    return pd.Series([sentence_toponyms, document_toponyms])


# ------------------------------
# Process data for training a NER model, where each sentence has an id,
# and a list of tokens and assigned ner_tags using the BIO scheme, e.g.:
# > id: 10813493_1 # document_id + "_" + sentence_id
# > ner_tags: ['B-LOCWiki', 'O']
# > tokens: ['INDIA', '.']
def process_for_ner(tsv_topres_path):
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
            {"id": file_id + "_" + str(prev_sent), "ner_tags": ner_tags, "tokens": tokens}
        )

    lwm_df = pd.DataFrame(lwm_data)

    return lwm_df


# ------------------------------
# Process data for performing entity linking, resulting in a dataframe with
# one toponym per row and its annotation and resolution in columns.
def process_for_linking(tsv_topres_path, output_path):
    # # Create the dataframe where we will store our annotated
    # data in a format that works better for us:
    df = pd.DataFrame(
        columns=[
            "mention_id",
            "sent_id",
            "article_id",
            "place",
            "decade",
            "prev_sentence",
            "current_sentence",
            "marked_sentence",
            "next_sentence",
            "mention",
            "place_class",
            "place_wikititle",
            "place_wqid",
            "start",
            "end",
        ]
    )

    # Populate the dataframe of toponyms and annotations:
    mention_counter = 0
    for fid in glob.glob(tsv_topres_path + "annotated_tsv/*"):

        filename = fid.split("/")[-1]  # Full document name
        file_id = filename.split("_")[0]  # Document id
        publ_place = filename.split("_")[1][:-8]  # Publication place
        publ_decade = filename[-8:-4]  # Publication decade

        # Dictionary that maps each token in a document with its
        # positional information and associated annotations:
        dMTokens, dTokens = process_tsv(tsv_topres_path + "annotated_tsv/" + filename)

        # Dictionary of reconstructed sentences:
        dSentences = reconstruct_sentences(dTokens)

        for k in dMTokens:
            # For each token, create the list that will become the dataframe
            # row, if the token has been annotated:
            row = create_lwmdf_row(
                dMTokens[k], file_id, publ_place, publ_decade, mention_counter, dSentences
            )
            # If a row has been created:
            if row:
                # Convert the row into a pd.Series:
                row = pd.Series(row, index=df.columns)
                # And append it to the main dataframe:
                df = pd.concat([df, row.to_frame().T], ignore_index=True)

        # Store the sentences as a json so that:
        # * The key is an index indicating the order of the sentence.
        # * The value is a list of two elements: the first element is the text of the sentence,
        # while the second element is the character position of the first character of the
        # sentence in the document.
        Path(output_path + "lwm_sentences/").mkdir(parents=True, exist_ok=True)
        with open(output_path + "lwm_sentences/" + filename + ".json", "w") as fp:
            json.dump(dSentences, fp)

    df[["sentence_toponyms", "document_toponyms"]] = df.apply(
        lambda x: pd.Series(add_cotoponyms(df, x["article_id"], x["sent_id"])), axis=1
    )

    return df


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
    with open("outputs/results/" + dataset + "/" + dataresults + "_bundle2_en_1.tsv", "w") as fw:
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
                fw.write(t[0] + "\t" + t[1] + "\t0\tO\tO\tO\tO\t" + elink + "\tO\tO\n")
            fw.write("\n")


def store_resolution_skyline(dataset, approach, value):
    pathlib.Path("outputs/results/" + dataset + "/").mkdir(parents=True, exist_ok=True)
    skyline = open("outputs/results/" + dataset + "/" + approach + ".skyline", "w")
    skyline.write(str(value))
    skyline.close()


def read_gold_standard(path):
    if Path(path).is_file():
        with open(path) as f:
            d = json.load(f)
            return d
    else:
        print("The tokenised gold standard is missing. You should first run the LwM baselines.")
        exit()


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
                el = urllib.parse.quote(ent[3].replace("_", " "))
                try:
                    el = ent_pos + wikipedia2wikidata[el]
                except Exception:
                    # to be checked but it seems some Wikipedia pages are not in our Wikidata
                    # see for instance Zante%2C%20California
                    return n, "O", ""
                return n, el, prev_ann
    return "O", "O", ""
