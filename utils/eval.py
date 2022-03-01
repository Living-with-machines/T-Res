
def eval_selection(true_mentions_sents,trues,preds):

    pres = 0
    tot = 0

    for id in true_mentions_sents.keys():
        true_sent = true_mentions_sents[id]
        for mention in true_sent:
            start_offset = mention['start_offset']
            end_offset = mention['end_offset']
            ent_id = trues[id][start_offset][2]
            if "B-Q" in ent_id:
                ent_id = ent_id[2:]
                pred_cand_ents = {}
                for x in range(start_offset,end_offset+1):
                    # if we have candidates from the candidate selection step
                    if len(preds[id][x])>3:
                        pred_cand_ents = pred_cand_ents | preds[id][x][3].keys()
                tot +=1
                if ent_id in pred_cand_ents:
                    pres +=1
    return pres/tot