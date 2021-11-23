#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import json
import pandas as pd
from pathlib import Path

def test_lwm_data_exists():
    assert Path("resources/topRes19th/").is_dir()
    assert Path("resources/wikipedia/test-extractedResources/").is_dir()
    assert Path("resources/wikipedia/test-extractedResources/entities_overall_dict.json").is_file()

def test_entity_dict():
    with open('resources/wikipedia/test-extractedResources/entities_overall_dict.json', 'r') as f:
        entities_overall_dict = json.load(f)
    assert len(entities_overall_dict)==4540

def test_lwm_dataframe_exists():
    assert Path("outputs/data/lwm_df.tsv").is_file()

def test_lwm_dataframe_notempty():    
    df = pd.read_csv("outputs/data/lwm_df.tsv", sep="\t")
    assert df.empty == False

def test_lwm_dataframe_shape():
    df = pd.read_csv("outputs/data/lwm_df.tsv", sep="\t")
    assert df.shape == (3348, 14)
    
def test_lwm_sentences_files():
    # The number of files with processed sentences is the same as the number of annotated documents:
    assert len([y for y in Path('outputs/data/lwm_sentences/').glob('*.json')]) == len([y for y in Path('resources/topRes19th/annotated_tsv/').glob('*.tsv')])  

