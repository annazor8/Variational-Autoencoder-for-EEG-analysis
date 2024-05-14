from pyeeglab import Preprocessor
import json
import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from hashlib import md5
from typing import List, Dict
from mne.io import Raw


DEBUG_MODE = False


class Pipeline():

    environment: Dict = {}
    pipeline: List[Preprocessor]

    def __init__(self, preprocessors: List[Preprocessor] = [], labels_mapping: Dict = None) -> None:
        logging.debug('Create new preprocessing pipeline')
        self.pipeline = preprocessors
        self.labels_mapping = labels_mapping

    def _check_nans(self, data):
        nans = False
        if isinstance(data, np.ndarray):
            nans = np.any(np.isnan(data))
        if isinstance(data, pd.DataFrame):
            nans = data.isnull().values.any()
        return nans
    #annotation is a list of tuples where the first element is d, an element of data a the index i and the second element is a dictionary thet merges two dictionares
    def _trigger_pipeline(self, annotation, kwargs):
        data = None
        
        with annotation as reader: # See method __enter__ of Annotation_artifact (return a Raw)
            data: Raw = reader.load_data()
        
        for preprocessor in self.pipeline:
            print(f"Executing {preprocessor.__class__.__name__} with input {type(data)} and output ...", end=" ")
            if preprocessor.__class__.__name__ == 'ToNumpy':
                print(data) # columns
                print("debug data here")
            data = preprocessor.run(data, **kwargs) # 
            print(f"...  {type(data)}")

        
        nans = False
        if isinstance(data, list):
            nans = any([self._check_nans(d) for d in data])
        else:
            nans = self._check_nans(data)
        if nans:
            logging.debug(annotation.file_uuid)
            #raise ValueError('Nans found in file with id {}'.format(annotation.file_uuid))

        return data
    # data in input a run è una lista
    def run(self, data_list: List) -> Dict:
        """ RUN pipeline multiprocess
        Args:
            data_list: list of Annotation_artifact
        Rerturn:
        {
        'data': data_out,
        'labels': labels,
        'labels_encoder': onehot_encoder,
        'ID' : sj_IDs}
        
        Return
        Notes :
        # We gave as input N=6 data_list of Annotation_artifact
        # returns a dict with the following keys:
        # data_out there are 6 np.ndarray (samples x 24 x 35) 
        #   with 24 being channesl and 35 being [n-channels]+[some-features]
        # labels: 6 (0 an 1 encoded as in 'labels_encoder')
        # ID: 6 string all equal to "dataset"
           """
        logging.debug('Environment variables: {}'.format(
            str(self.environment)
        ))
        #data[0]=Annotation_artifact(uuid='cca940a6-fbaa-4b3e-84d9-eda214dc46fb', file_uuid='7da24417-7268-5076-a7fd-e5bf4113601f', begin=0.0, end=299.9, 
        #label='artifact', sj_ID='data1', sj_sess='aaaaaprj_s002_t000', mean_value=3.815641996972407, std_value=71.81454408510791, interval=[(19.8014, 46.2949), (98.877, 109.5569), (110.0136, 111.6971), (165.2422, 169.6883), (222.8648, 227.3402)], segment=23.0)
        labels = [raw.label for raw in data_list]
        sj_IDs = [raw.sj_ID for raw in data_list]
        #val.sj_ID è sempre data1, potrrebbe essere un problema
        #new_val_dict è una lista di dizionari che contiene per ogni soggetto due dizionari: uno per gli intervalli artefattuali e uno per gli intervalli del segnale pulito 
        new_val_dict = [{'ID':val.sj_ID,
                        'interval': [x for x in val.interval],
                        'mean': val.mean_value, 'std': val.std_value} 
                        for i, val in enumerate(data_list)]
        
        env = self.environment
        #data_list is a list of tuples where the first element is d, an element of data a the index i and the second element is a dictionary thet merges two dictionares
        if DEBUG_MODE:
            # data_out = []
            # for i, data_annotation in enumerate(data_list):
            #     temp_out = self._trigger_pipeline(annotation=data_annotation, kwargs={**env,**new_val_dict[i]})
            #     data_out.append(temp_out)
            data = [(d, {**env,**new_val_dict[i]}) for i, d in enumerate(data_list)]
            pool = Pool(1)
            #starmap takes an iterable of argument tuples, where each tuple contains the arguments passed to the function.
            #starmap will unpack each tuple in data and pass them as separate arguments to self._trigger_pipeline
            data_out = pool.starmap(self._trigger_pipeline, data)
        else:
            data = [(d, {**env,**new_val_dict[i]}) for i, d in enumerate(data_list)]
            pool = Pool(cpu_count())
            #starmap takes an iterable of argument tuples, where each tuple contains the arguments passed to the function.
            #starmap will unpack each tuple in data and pass them as separate arguments to self._trigger_pipeline
            data_out = pool.starmap(self._trigger_pipeline, data)
        
        #the output data is a list containing the results of the self._trigger_pipeline method executed on each tuple of the original data
        pool.close()
        pool.join()
        if self.labels_mapping is not None:
            labels = [self.labels_mapping[label] for label in labels]
        onehot_encoder = sorted(set(labels))
        class_id = self.environment.get('class_id', None)
        if class_id in onehot_encoder:
            onehot_encoder.remove(class_id)
            onehot_encoder = [class_id] + onehot_encoder
        labels = np.array([onehot_encoder.index(label) for label in labels])

        # # The following is commented because it gives error in case the pipeline outputs 
        # a 3D array with different shapes (our case so far)

        # if any([p.__class__.__name__ == 'ToNumpy' for p in self.pipeline]):
        #     data_out = np.array(data_out)
        

        return {'data': data_out, 'labels': labels, 'labels_encoder': onehot_encoder, 'ID' : sj_IDs}

    def to_json(self) -> str:
        json = [p.to_json() for p in self.pipeline]
        json = '[ ' + ', '.join(json) + ' ]'
        return json

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        value = [p.to_json() for p in self.pipeline]
        value = json.dumps(value).encode()
        value = md5(value).hexdigest()
        value = int(value, 16)
        return value
