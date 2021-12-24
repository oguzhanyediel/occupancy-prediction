#!/usr/bin/python3
# -*- coding: utf-8 -*-

from prophet import Prophet
from prophet.serialize import model_to_json

import json
import logging
import os
import sys

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Occupancy Prediction - Model File')

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))


def fit_predict_model(model_data, device_model_name):
    """
    :param model_data: dataframe
    :param device_model_name: string
    :return: None
    """
    try:
        # define the model
        m = Prophet()
        # fit the model
        m.fit(model_data)
        # save the model
        with open(model_path + '/serialized_model_{0}.json'.format(device_model_name), 'w') as fout:
            json.dump(model_to_json(m), fout)
    except Exception as e:
        logger.error(e.args)
        sys.exit(1)
