import sys,os
# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath('CLEF-HIPE-2020-scorer/'))

import clef_evaluation

dataset = 'lwm'
approach = 'perfectmatch+mostpopular'

pred = '../experiments/outputs/results/'+dataset+'/'+approach+'_bundle2_en_1.tsv'
true = '../experiments/outputs/results/'+dataset+'/true_bundle2_en_1.tsv'

ner_score = clef_evaluation.get_results(f_ref=true,f_pred=pred,task='nerc_coarse')
print (ner_score['NE-COARSE-LIT']['TIME-ALL']['LED-ALL']['ALL']['strict']['F1_micro'])

skyline = open('../experiments/outputs/results/'+dataset+'/'+approach+'.skyline','r').read()
print (skyline)

linking_score = clef_evaluation.get_results(f_ref=true,f_pred=pred,task='nel')
print (linking_score[1]['NEL-LIT']['TIME-ALL']['LED-ALL']['ALL']['strict']['F1_micro'])