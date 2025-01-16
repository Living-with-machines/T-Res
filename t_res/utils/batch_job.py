import os
import sys
import yaml
import json
import argparse
import logging
import pickle
from math import ceil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

import pandas as pd
import sqlite3

from t_res.geoparser import ner, ranking, linking, pipeline

RECOGNISER_KEY = 'recogniser'
RANKER_KEY = 'ranker'
LINKER_KEY = 'linker'
BATCH_SIZE_KEY = 'batch_size'

def run():
    parser = argparse.ArgumentParser(description='Run a T-Res batch job.')
    parser.add_argument('config_file', type=str, help='Path to the YAML batch job config file.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV data file.')
    parser.add_argument('resources_path', type=str, help='Path to the resources directory.')
    parser.add_argument('results_path', type=str, help='Path to the results directory.')
    help = '''[Optional] Path to the place of publication CSV data file. \
        Must include columns named "Wikidata ID" and "Location"'''
    parser.add_argument('place_of_pub_file', type=str, help=help, nargs='?')

    args = parser.parse_args()

    with open(args.config_file) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            print(f"Error parsing YAML config file: {err}")
            sys.exit()

    validate_config(config)
    tqdm.pandas()

    if not os.path.exists(args.resources_path):
        raise ValueError(f"Resources path does not exist: {args.resources_path}")
    if not os.path.isfile(args.input_file):
        raise ValueError(f"Missing input data file: {args.input_file}")
    Path(args.results_path).mkdir(parents=True, exist_ok=True)
    
    batch_job = BatchJob.new(
        batch_size=config[BATCH_SIZE_KEY],
        config=config, 
        input_file=args.input_file,
        resources_path=args.resources_path,
        results_path=args.results_path,
        place_of_pub_file=args.place_of_pub_file,
    )
    batch_job.load()
    batch_job.run()

def validate_config(config: dict):
    keys = {RECOGNISER_KEY, RANKER_KEY, LINKER_KEY, BATCH_SIZE_KEY}
    missing_keys = keys.difference(config.keys())
    if missing_keys:
        raise ValueError(f"Missing config key(s): {missing_keys}")

class BatchJob:
    """
    A wrapper for the Pipeline class for efficient & convenient processing 
    of large datasets.
    """

    predictions_pickle = 'predictions.pkl'

    place_of_pub_wqid_key = 'place_of_pub_wqid'
    place_of_pub_key = 'place_of_pub'

    def __init__(
        self,
        config: dict,
        input_file: str,
        resources_path: str,
        results_path: str,
        place_of_pub_file: Optional[str]=None,
    ):

        self.config = config
        self.config_str = json.dumps(config, indent=4)
        self.resources_path = resources_path
        self.results_path = results_path
        self.input_file = input_file
        self.place_of_pub_file = place_of_pub_file

        try:
            self.batch_size = int(config[BATCH_SIZE_KEY])
        except:
            raise ValueError(f'Batch size must be an integer. Use 0 for unlimited batch size.')

        self.batches_processed = 0

    def new(batch_size: int, **kwargs) -> 'BatchJob':
        """
        Static constructor.

        Args:
            batch_size (int): A non-negative integer. The size of each batch.
            kwargs (dict): A dictionary of keyword arguments matching the
                arguments to the BatchJob __init__ constructor.

        Returns:
            A BatchJob (subclass) instance.
        """
        if batch_size == 0:
            return UnlimitedBatchJob(**kwargs)
        if batch_size == 1:
            return SingletonBatchJob(**kwargs)
        if batch_size > 1:
            return LimitedBatchJob(**kwargs)
        raise ValueError(f"Invalid batch_size: {batch_size}")

    def load(self):

        # Construct the T-Res pipeline.
        self.construct_pipeline()

        # Read input data & drop rows with empty text.
        self.input_data = pd.read_csv(self.input_file).dropna(subset=["text"])

        # Read place of publication information into a dictionary.
        if self.place_of_pub_file:

            place_of_pub_data = {}
            for i, row in pd.read_csv(self.place_of_pub_file).iterrows():
                place_of_pub_data[row["NLP"]] = {
                    self.place_of_pub_wqid_key: row["Wikidata ID"], 
                    self.place_of_pub_key: row["location"]
                }
            self.place_of_pub_data = place_of_pub_data

            self.place_of_pub_series = self.input_data.apply(
                lambda x: place_of_pub_data[x["NLP"]],
                axis=1,
            )

        else:
            self.place_of_pub_data = None

    def construct_pipeline(self):

        recogniser = ner.Recogniser.new(
            **self.config[RECOGNISER_KEY])
        ranker = ranking.Ranker.new(
            resources_path=self.resources_path, 
            **self.config[RANKER_KEY])
        
        # Fill in linking parameters in the case of a REL Linker.
        if self.config[LINKER_KEY]['method_name'] == 'reldisamb':
            self.config[LINKER_KEY]['ranker'] = ranker
            rel_params = self.config[LINKER_KEY]['rel_params']
            rel_params['do_test'] = False
            rel_params['model_path'] = os.path.join(self.resources_path, "models/disambiguation/")
            db_database_path = os.path.join(self.resources_path, "rel_db/embeddings_database.db")
            with sqlite3.connect(db_database_path) as conn:
                rel_params['db_embeddings'] = conn.cursor()

        linker = linking.Linker.new(
            resources_path=self.resources_path,
            **self.config[LINKER_KEY])

        self.pipe = pipeline.Pipeline(
            recogniser=recogniser,
            ranker=ranker,
            linker=linker,
        )
        # self.logger.info('Constructed T-Res pipeline')

    def initialise_logging(self):

        logger = logging.getLogger(__name__)
        self.log_file = os.path.join(self.run_path, f'{self.run_title()}.log')
        logging.basicConfig(
            filename=self.log_file, 
            encoding='utf-8', 
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
        )
        self.logger = logger

        print(">>>> Running T-Res batch job >>>>")
        self.logger.info(f'Starting T-Res batch job...')
        self.logger.info(f'Input data file: {self.input_file}')
        if self.place_of_pub_file:
            self.logger.info(f'Place of publication data file: {self.place_of_pub_file}')
        self.logger.info(f'Results will be written to: {self.results_path}')
        self.logger.info(f'Resources will be read from: {self.resources_path}')
        self.logger.info(f'Config:\n{self.config_str}')

    def timestamp(self) -> str:
        return self.start_time.strftime('%Y-%m-%d_%H-%M-%S')

    def run_title(self) -> str:
        return f't-res_batch_{self.timestamp()}'

    def run(self):

        # Store the start time of this run.
        self.start_time = datetime.now()

        # Create a subdirectory for this run.
        self.run_path = os.path.join(self.results_path, self.run_title())
        os.mkdir(self.run_path)
        self.initialise_logging()

        predictions = self.run_batches()

        # Store the start time of this run.
        self.end_time = datetime.now()

        # Save the predictions.
        with open(os.path.join(self.run_path, self.predictions_pickle), 'wb') as f:
            pickle.dump(predictions, f)

        # TODO NEXT: Save main results in a copy of the input spreadsheet
        # Decide how those results should look (see UK-France sample results for a hint but try to improve)
        self.save_results()

        # TODO: tidy up (remove intermediate files, unless configured to keep them).
        self.logger.info('Batch job finished successfully.')
        self.logger.info(f'Execution time: {self.execution_time()}')
        print(f'>>>> T-Res batch job finished ({self.execution_time()}) <<<<')

    def run_batches(self) -> pd.Series:

        predictions_list = list()
        while(self.next_batch_range()):
            # Split input into batches of size batch_size.
            r = self.next_batch_range()
            next_batch = self.input_data.iloc[r[0]:r[1]]
            self.logger.info(f'Running batch {self.batches_processed + 1}. Items {r[0]}-{r[1]}')
            print(f'Batch {self.batches_processed + 1} of {self.count_batches()}:')
            predictions_list.append(self.run_batch(next_batch))
        return pd.concat(predictions_list)

    def run_batch(self, batch) -> pd.Series:

        mentions_series = self.run_batch_ner(batch)
        candidates_series = self.run_batch_ranking(mentions_series)
        predictions_series = self.run_batch_linking(candidates_series)
        self.batches_processed += 1
        return predictions_series

    def run_batch_ner(self, batch) -> pd.Series:

        # Define function to discard sentences not containing toponym mentions.
        def ner_or_none(text):
            mentions = self.pipe.run_text_recognition(text)
            return [sm for sm in mentions if not sm.is_empty()]

        print('NER...')
        tick = datetime.now()
        result = batch.progress_apply(
            lambda x: ner_or_none(x["text"]),
            axis=1,
        )
        tock = datetime.now() 
        self.logger.info(f'NER execution time: {tock - tick}')
        return result
    
    def run_batch_ranking(self, mentions_series) -> pd.Series:

        print('Candidate selection...')
        tick = datetime.now()
        # Convert to a data frame to access the row index via the `name` field.
        result = pd.DataFrame(mentions_series).progress_apply(
            lambda x: self.pipe.run_candidate_selection(
                x[0],
                place_of_pub_wqid=self.place_of_pub_series[x.name][self.place_of_pub_wqid_key],
                place_of_pub=self.place_of_pub_series[x.name][self.place_of_pub_key],
            ),
            axis=1,
        )
        tock = datetime.now() 
        self.logger.info(f'Candidate Selection execution time: {tock - tick}')
        return result

    def run_batch_linking(self, candidate_series):

        print('Disambiguation...')
        tick = datetime.now()
        result = candidate_series.progress_apply(
            lambda x: self.pipe.run_disambiguation(x), 
        )
        tock = datetime.now() 
        self.logger.info(f'Disambiguation execution time: {tock - tick}')
        return result

    def save_results(self):

        results_str = ''
        # TODO.
        return

    def next_batch_range(self) -> Optional[Tuple[int, int]]:
        """
        Computes the range of indices of items in the next batch.
        Item indices start counting from zero and the range is inclusive
        of the lower end of the range and exclusive of the upper end.

        For instance, if batch range (0, 10) includes items with indices
        0 to 9.

        Returns:
            The range of item indices in the next batch.
        """

        # Count items from zero.
        next_item = self.batches_processed * self.batch_size
        final_item = len(self.input_data.index) - 1
        if next_item > final_item:
            return None
        return next_item, min(next_item + self.batch_size, final_item + 1)
    
    def count_batches(self) -> int:
        return ceil(len(self.input_data.index) / self.batch_size)

    def execution_time(self):
        return self.end_time - self.start_time
    
class LimitedBatchJob(BatchJob):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.batch_size > 0:
            raise ValueError(f'Invalid batch size: {self.batch_size}')

class UnlimitedBatchJob(BatchJob):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.batch_size != 0:
            raise ValueError(f'Invalid batch size: {self.batch_size}')

    # Override the `load` method to set the batch size equal to the input data size.
    def load(self):
        super().load()
        self.batch_size = len(self.input_data.index)

class SingletonBatchJob(BatchJob):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.batch_size != 1:
            raise ValueError(f'Invalid batch size: {self.batch_size}')
        
    # Override the `run_batches` method to run the pipeline end-to-end.
    def run_batches(self) -> pd.Series:

        print('Running end-to-end pipeline...')
        return self.input_data.progress_apply(
            lambda x: self.pipe.run(x["text"]),
            axis=1,
        )

