#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pytest

def test_lwm_data_exists():
    from pathlib import Path
    assert Path("resources/topRes19th/").is_dir()

def test_lwm_dataframe_exists():
    from pathlib import Path
    assert Path("outputs/data/lwm_df.tsv").is_file()

def test_lwm_dataframe_notempty():
    import pandas as pd
    df = pd.read_csv("outputs/data/lwm_df.tsv", sep="\t")
    assert df.empty == False

def test_lwm_dataframe_shape():
    import pandas as pd
    df = pd.read_csv("outputs/data/lwm_df.tsv", sep="\t")
    assert df.shape == (3348, 12)