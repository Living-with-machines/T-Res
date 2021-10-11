#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pytest

def test_lwm_data():
    from pathlib import Path
    assert Path("resources/topRes19th/").is_dir()
