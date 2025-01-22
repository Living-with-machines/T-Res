import pytest
import os

import pandas as pd
from t_res.utils.batch_job import *
from t_res.utils.dataclasses import Predictions

current_dir = Path(__file__).parent.resolve()

def sample_config_basic():
    return {
        'recogniser': {
            'method_name': 'pretrained', 
            'model_name': 'Livingwithmachines/toponym-19thC-en'},
        'ranker': {'method_name': 'perfectmatch'},
        'linker': {'method_name': 'mostpopular'},
        'batch_size': 20
    }

def sample_batch_job(config, tmp_path):
    input_file = os.path.join(current_dir, './sample_files/batch_jobs/1880-1900-LwM-HMD-subsample50.csv')
    resources_path = os.path.join(current_dir, '../resources/')
    place_of_pub_file = os.path.join(current_dir, './sample_files/batch_jobs/newspapers_wikidata_ids.csv')
    return BatchJob.new(
        batch_size=config['batch_size'],
        config=config, 
        input_file=input_file,
        resources_path=resources_path,
        results_path=tmp_path,
        place_of_pub_file=place_of_pub_file,
    )

def test_static_constructor():
    config = sample_config_basic()
    input_file = os.path.join(current_dir, './sample_files/batch_jobs/1880-1900-LwM-HMD-subsample50.csv')
    resources_path = os.path.join(current_dir, '../resources/')
    results_path = os.path.join(current_dir, '../results/')

    config['batch_size'] = 0
    batch_job = BatchJob.new(
        batch_size=config[BATCH_SIZE_KEY],
        config=config, 
        input_file=input_file,
        resources_path=resources_path,
        results_path=results_path,
    )
    assert isinstance(batch_job, UnlimitedBatchJob)
    assert batch_job.batch_size == 0

    config['batch_size'] = 1
    batch_job = BatchJob.new(
        batch_size=config[BATCH_SIZE_KEY],
        config=config, 
        input_file=input_file,
        resources_path=resources_path,
        results_path=results_path,
    )
    assert isinstance(batch_job, SingletonBatchJob)
    assert batch_job.batch_size == 1

    config['batch_size'] = 10
    batch_job = BatchJob.new(
        batch_size=config[BATCH_SIZE_KEY],
        config=config, 
        input_file=input_file,
        resources_path=resources_path,
        results_path=results_path,
    )
    assert isinstance(batch_job, LimitedBatchJob)
    assert batch_job.batch_size == 10

@pytest.mark.resources(reason="Needs large resources")
def test_next_batch_range(tmp_path):
    config = sample_config_basic()

    batch_job = sample_batch_job(config, tmp_path)
    isinstance(batch_job, LimitedBatchJob)
    batch_job.load()

    assert len(batch_job.input_data.index) == 49

    assert batch_job.batches_processed == 0
    assert batch_job.next_batch_range() == (0, 20)
    batch_job.batches_processed = 1
    assert batch_job.next_batch_range() == (20, 40)
    batch_job.batches_processed = 2
    assert batch_job.next_batch_range() == (40, 49)
    batch_job.batches_processed = 3
    assert batch_job.next_batch_range() is None

    config['batch_size'] = 0

    batch_job = sample_batch_job(config, tmp_path)
    isinstance(batch_job, UnlimitedBatchJob)
    assert batch_job.batch_size == 0
    batch_job.load()
    assert len(batch_job.input_data.index) == 49
    assert batch_job.batch_size == 49

    assert batch_job.batches_processed == 0
    assert batch_job.next_batch_range() == (0, 49)

    batch_job.batches_processed = 1
    assert batch_job.next_batch_range() is None

@pytest.mark.resources(reason="Needs large resources")
def test_run_batch_job(tmp_path):

    config = sample_config_basic()

    batch_job = sample_batch_job(config, tmp_path)
    batch_job.load()
    
    batch_job.run()

    # assert os.path.isdir(os.path.join(tmp_path, batch_job.run_path))
    assert os.path.isdir(os.path.join(batch_job.run_path))

    # assert os.path.isfile(batch_job.log_file)

    # Check the pickled results.
    pickle_file = os.path.join(batch_job.run_path, BatchJob.predictions_pickle)
    assert os.path.isfile(pickle_file)

    with open(pickle_file, 'rb') as f:
        predictions = pickle.load(f)
    assert isinstance(predictions, pd.Series)
    assert predictions.size == 49

    assert isinstance(predictions[0], Predictions)

    # First row in the input data is NLP 3406: "Nantwich, Cheshire, England", Q1077003
    assert predictions[0].place_of_pub_wqid() == "Q1077003"
    assert predictions[0].place_of_pub() == "Nantwich, Cheshire, England"

    # Last row in the input data is NLP 3406: "Nantwich, Cheshire, England", Q1077003
    assert predictions[0].place_of_pub_wqid() == "Q1077003"
    assert predictions[0].place_of_pub() == "Nantwich, Cheshire, England"

    # Check the CSV results.
    assert os.path.isfile(batch_job.results_file())
    with open(batch_job.results_file(), 'r') as f:
        results = pd.read_csv(f)

    assert list(results.columns)[-1] == batch_job.predictions_colname
    print(results)
