import pandas as pd
import ast
from difflib import SequenceMatcher
import random
from pathlib import Path
from tqdm import tqdm

# Parameters:
length_diff = 2 # accepted length difference between OCR and correction
len_token = [5,15] # minimal and maximal length token
chall_thr = [0.4, 0.7] # min and max threshold (in str sim) for challenging matches
max_patience = 10000 # Break if no negative match is found after {max_patience} random comparisons
num_chall = 10 # 1/{num_chall} is the ratio of challenging negative pairs (i.e. similar)

# Dataframe with aligned trove data: # TO DO: make aligned dataset available
df = pd.read_pickle("/resources/develop/mcollardanuy/toponym-resolution/resources/ocr/trove_subsample_aligned.pkl")

# Path where we will store the resulting toponym pairs dataset:
output_path = "experiments/datasets/" # Path for DeezyMatch hyperparameter experiments
# output_path = "../../experiments/outputs/deezymatch/datasets/" # Path for final experiments

Path(output_path).mkdir(parents=True, exist_ok=True)


def acceptable(a, b, threshold, len_token):
    # a: OCR token
    # b: human-corrected token
    # thrshold: minimal string similarity value
    # len_token: min and max length of OCR token to be considered
    value = SequenceMatcher(None, a, b).ratio()
    if len_token[0] < len(a) < len_token[1]:
        if len_token[0] < len(b) < len_token[1]:
            if abs(len(a) - len(b)) <= 2:
                # length difference is of 2 characters:
                if (a == b or ((value >= threshold) and (not a in b) and (not b in a))):
                    # a is not contained in b, b is not contained in a:
                    return True
    else:
        return False
    
    
# Align the OCR token with its human-corrected counterpart if they 
# are acceptable (longer than x, more similar than y, a not contained
# in b, b not contained in a, or a equals b)
human_to_ocr = dict()
for i, row in df.iterrows():
    ocrText = row['ocrText']
    humanText = row['corrected']
    alignments = ast.literal_eval(row['alignment'])
    for a in alignments:
        ocr_token = ocrText[a[0]:a[1]]
        human_token = humanText[a[2]:a[3]]
        if acceptable(ocr_token, human_token, chall_thr[1], len_token):
            if human_token in human_to_ocr:
                human_to_ocr[human_token].append(ocr_token)
            else:
                human_to_ocr[human_token] = [ocr_token]
                
                
for x in human_to_ocr:
    # Artificially add s<->f confusion:
    if "s" in x[:-1]:
        human_to_ocr[x].append(x[:-1].replace("s", "f", 1) + x[-1])
    # Artificially add white space, hyphen or dot (1/100) at a random position:
    spc = [" ", ".", "-"] + [""] * 100
    randomch = random.choice(spc)
    randompos = random.randint(0, len(x) - 1)
    newx = x[:randompos] + randomch + x[randompos:]
    human_to_ocr[x].append(newx)
    
    
# List of unique OCRed tokens in our pairs:
all_ocr_tokens = [human_to_ocr[x] for x in human_to_ocr]
all_ocr_tokens = [item for sublist in all_ocr_tokens for item in sublist]
all_ocr_tokens = list(set(all_ocr_tokens))


# For each true match, we create an artificial either trivial or challenging negative match:
true_pairs = []
false_pairs = []
count_true_matches = 0
count_false_matches = 0
for k in tqdm(human_to_ocr):
    for h in human_to_ocr[k]:
        true_pairs.append((k, h, "TRUE"))
        count_true_matches += 1
    for i in range(len(human_to_ocr[k])):
        found_negative_match = False
        # Challenging negative matches:
        if random.randint(0,num_chall) == 0:
            counter_rand = 0
            while found_negative_match == False:
                counter_rand += 1
                # Break if no match is found after {max_patience} random comparisons:
                if counter_rand >= max_patience:
                    break
                random_one = random.choice(all_ocr_tokens)
                if abs(len(random_one) - len(k)) <= length_diff:
                    # Challenging match (similarity between min and max thr):
                    if SequenceMatcher(None, random_one, k).ratio() < chall_thr[1] and SequenceMatcher(None, random_one, k).ratio() > chall_thr[0]:
                        false_pairs.append((random_one, k, "FALSE"))
                        found_negative_match = True
                        count_false_matches += 1
        else:
            while found_negative_match == False:
                random_one = random.choice(all_ocr_tokens)
                if abs(len(random_one) - len(k)) <= length_diff:
                    # Trivial match (similarity below min thr):
                    if SequenceMatcher(None, random_one, k).ratio() < chall_thr[0]:
                        false_pairs.append((random_one, k, "FALSE"))
                        found_negative_match = True
                        count_false_matches += 1
                        
                
random.shuffle(true_pairs)
random.shuffle(false_pairs)


Path(output_path).mkdir(parents=True, exist_ok=True)

# Store the dataset in the format required by DeezyMatch
with open(output_path + "ocr_string_pairs.txt", "w") as fw:
    for t in true_pairs:
        fw.write(t[0] + "\t" + t[1] + "\t" + t[2] + "\n")
    for f in false_pairs:
        fw.write(f[0] + "\t" + f[1] + "\t" + f[2] + "\n")