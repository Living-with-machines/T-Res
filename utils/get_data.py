import re
import wget
import pathlib
import zipfile
import pandas as pd


# ------------------------------
# Download data from BL repository and unzip it.
# Output will be stored in resources/
def download_lwm_data():
    url = "https://bl.iro.bl.uk/downloads/ff44881f-97ca-4c68-97f8-097324bdba94?locale=en"
    save_to = "resources"
    pathlib.Path(save_to).mkdir(parents=True, exist_ok=True)
    if not pathlib.Path("resources/topRes19th.zip").is_file():
        lwm_dataset = wget.download(url, out=save_to)
        with zipfile.ZipFile(lwm_dataset) as zip_ref:
            zip_ref.extractall(save_to)
            

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
    regex_annline = r'^[0-9]+\-[0-9]+\t[0-9]+\-[0-9]+\t.*$'
    
    # This regex identifies annotations that span multiple tokens:
    regex_multmention = r'^(.*)\[([0-9]+)\]'

    multiple_mention = 0
    prev_multmention = 0
    complete_token = ""
    complete_label = ""
    complete_wkpd = ""
    dTokens = dict()
    sent_pos = 0
    tok_pos = 0
    tok_start = 0
    tok_end = 0
    
    # Loop over all lines in the file:
    for line in lines:
        
        # If the line is a token-line:
        if re.match(regex_annline, line):
            
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
                
            # If the annotation corresponds to a multi-token annotation:
            if re.match(regex_multmention, wkpd):
                
                # This code basically collates multi-token mentions in annotations
                # together. "complete_token" is the resulting multi-token mention,
                # "sent_pos" is the sentence position in the file, "tok_pos" is the
                # token position in the sentence, "tok_start" is the multi-token
                # character start position in the document, and "tok_end" is the
                # multi-token character end position in the document.
                multiple_mention = int(re.match(regex_multmention, wkpd).group(2))
                complete_label = re.match(regex_multmention, label).group(1)
                complete_wkpd = re.match(regex_multmention, wkpd).group(1)
                
                if multiple_mention == prev_multmention:
                    complete_token += " " + token
                    tok_end = tok_tmp.split("-")[1]
                else:
                    sent_pos, tok_pos = sent_tmp.split("-")
                    tok_start, tok_end = tok_tmp.split("-")
                    complete_token = token
                prev_multmention = multiple_mention

            # If the annotation does not correspond to a multi-token annotation,
            # just keep the token-specific information:
            else:
                sent_pos, tok_pos = sent_tmp.split("-")
                tok_start, tok_end = tok_tmp.split("-")
                complete_token = token
                complete_label = label
                complete_wkpd = wkpd

            sent_pos = int(sent_pos)
            tok_pos = int(tok_pos)
            tok_start = int(tok_start)
            tok_end = int(tok_end)

            dTokens[(sent_pos, tok_start)] = (complete_token, complete_wkpd, complete_label, sent_pos, tok_start, tok_end)
    
    return dTokens


# ------------------------------
# Given the dictionary of tokens (with their positional information
# in the document and associated annotations), reconstruct all sentences
# in the document (taking into account white spaces to make sure character
# positions match).
def reconstruct_sentences(dTokens):
    
    complete_sentence = "" # In this variable we keep each complete sentence
    sentence_id = 0 # In this variable we keep the sentence id
    start_ids = [k[1] for k in dTokens] # Ordered list of token character start positions.
    dSentences = dict() # In this variable we'll map the sentence id with the complete sentence
    prev_sentid = 0 # Keeping track of previous sentence id (to know when a new sentence is starting)
    dSentenceCharstart = dict() # In this variable we will map the sentence id with the start character
                                # position of the sentence in question
    
    dIndices = dict() # In this dictionary we map the token start character in the document with 
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
# Populate dataframe rows:
def create_lwmdf_row(mention_values, file_id, publ_place, publ_decade, mention_counter, dSentences):

    mention, wkpd, label, sent_pos, tok_start, tok_end = mention_values
    if not wkpd == "_" and not label == "_":
        mention_counter += 1
        current_sentence, sentence_start = dSentences[sent_pos]
        mention_start = tok_start - sentence_start
        marked_sentence = current_sentence
        marked_sentence = marked_sentence[:mention_start] + " [MASK] " + marked_sentence[mention_start + len(mention):]
        marked_sentence = re.sub(' +', ' ', marked_sentence)
        prev_sentence = ""
        if sent_pos-1 in dSentences:
            prev_sentence = dSentences[sent_pos-1][0]
        next_sentence = ""
        if sent_pos+1 in dSentences:
            next_sentence = dSentences[sent_pos+1][0]

        row = [mention_counter, sent_pos, file_id, publ_place, publ_decade, prev_sentence, current_sentence, marked_sentence, next_sentence, mention, label, wkpd]
        
        return row
    
    return False