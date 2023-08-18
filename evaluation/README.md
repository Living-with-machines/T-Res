# Evaluation

First, clone the [CLEF-HIPE-2020-scorer](https://github.com/impresso/CLEF-HIPE-2020-scorer) to this folder and checkout [this commit](https://github.com/impresso/CLEF-HIPE-2020-scorer/tree/ac5c876eba58065195024cff550c2b5056986f7b) to have the exact same evaluation setting as in our experiments.

```
git clone https://github.com/impresso/CLEF-HIPE-2020-scorer.git
cd HIPE-scorer
git checkout ac5c876eba58065195024cff550c2b5056986f7b
```

Then, to run the script:

To assess the performance on toponym recognition:
```bash
python HIPE-scorer/clef_evaluation.py --ref ../experiments/outputs/results/lwm-true_bundle2_en_1.tsv --pred ../experiments/outputs/results/lwm-pred_bundle2_en_1.tsv --task nerc_coarse --outdir results/
```

To assess the performance on toponym resolution:
```bash
python HIPE-scorer/clef_evaluation.py --ref ../experiments/outputs/results/lwm-true_bundle2_en_1.tsv --pred ../experiments/outputs/results/lwm-pred_bundle2_en_1.tsv --task nel --outdir results/
```
