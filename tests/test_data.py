#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import json
from pathlib import Path

def test_lwm_data_exists():
    assert Path("resources/topRes19th/").is_dir()
    assert Path("resources/wikipedia/test-extractedResources/").is_dir()
    assert Path("resources/wikipedia/test-extractedResources/entities_overall_dict.json").is_file()

def test_entity_dict():
    with open('resources/wikipedia/test-extractedResources/entities_overall_dict.json', 'r') as f:
        entities_overall_dict = json.load(f)
    assert len(entities_overall_dict)==4540