import sys,os
# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath('CLEF-HIPE-2020-scorer/'))

import clef_evaluation

dataset = 'lwm'

# Approach:
ner_model_id = 'rel'
cand_select_method = 'rel' # either perfectmatch or deezymatch
top_res_method = 'rel'

approach = ner_model_id+'+'+cand_select_method+'+'+top_res_method
print (approach)
pred = '../experiments/outputs/results/'+dataset+'/'+approach+'_bundle2_en_1.tsv'
true = '../experiments/outputs/results/'+dataset+'/true_bundle2_en_1.tsv'

metric_avg = 'macro'

for setting in ['strict','exact','partial']:
    print (setting)
    ner_score = clef_evaluation.get_results(f_ref=true,f_pred=pred,task='nerc_coarse',outdir='results/')
    print ('ner F1 '+metric_avg,round(ner_score['NE-COARSE-LIT']['TIME-ALL']['LED-ALL']['ALL'][setting]['F1_'+metric_avg],3))

    #skyline = open('../experiments/outputs/results/'+dataset+'/'+approach+'.skyline','r').read()
    #print (skyline)
    if setting == 'strict':
        linking_score = clef_evaluation.get_results(f_ref=true,f_pred=pred,task='nel',outdir='results/')
        print ('el F1 '+metric_avg,round(linking_score[1]['NEL-LIT']['TIME-ALL']['LED-ALL']['ALL'][setting]['F1_'+metric_avg],3))