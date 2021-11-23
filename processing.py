import glob
import json
import pandas as pd
from pathlib import Path
from utils import get_data
from utils import process_data

# Path for the output dataset dataframes:
output_path = "outputs/data/"

# ------------------------------------------------------
# LWM dataset
# ------------------------------------------------------

# Download the annotated data from the BL repository:
get_data.download_lwm_data()

# Path of the annotated data:
tsv_topres_path = "resources/topRes19th/annotated_tsv/"

# Create the dataframe where we will store our annotated
# data in a format that works better for us:
df = pd.DataFrame(columns = ["mention_id", "sent_id", "article_id", "place", "decade", "prev_sentence", "current_sentence", "marked_sentence", "next_sentence", "mention", "place_class", "place_wikiid"])

# Populate the dataframe of toponyms and annotations:
mention_counter = 0
for fid in glob.glob(tsv_topres_path + "*"):
    
    filename = fid.split("/")[-1] # Full document name
    file_id = filename.split("_")[0] # Document id
    publ_place = filename.split("_")[1][:-8] # Publication place
    publ_decade = filename[-8:-4] # Publication decade
    
    # Dictionary that maps each token in a document with its
    # positional information and associated annotations:
    dTokens = process_data.process_tsv(tsv_topres_path + filename)
    
    # Dictionary of reconstructed sentences:
    dSentences = process_data.reconstruct_sentences(dTokens)
    
    for k in dTokens:
        # For each token, create the list that will become the dataframe
        # row, if the token has been annotated:
        row = process_data.create_lwmdf_row(dTokens[k], file_id, publ_place, publ_decade, mention_counter, dSentences)
        # If a row has been created:
        if row:
            # Convert the row into a pd.Series:
            row = pd.Series(row, index=df.columns)
            # And append it to the main dataframe:
            df = df.append(row, ignore_index=True)
    
    # Store the sentences as a json so that:
    # * The key is an index indicating the order of the sentence.
    # * The value is a list of two elements: the first element is the text of the sentence,
    # while the second element is the character position of the first character of the
    # sentence in the document.
    Path(output_path + "lwm_sentences/").mkdir(parents=True, exist_ok=True)
    with open(output_path + "lwm_sentences/" + filename + '.json', 'w') as fp:
        json.dump(dSentences, fp)
        
df[['sentence_toponyms', 'document_toponyms']] = df.apply(lambda x: pd.Series(process_data.add_cotoponyms(df, x["article_id"], x["sent_id"])), axis=1)

Path(output_path).mkdir(parents=True, exist_ok=True)
df.to_csv(output_path + "lwm_df.tsv", sep="\t", index=False)

print(df.shape)