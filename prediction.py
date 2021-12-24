#!/usr/bin/python3
# -*- coding: utf-8 -*-

from prophet.serialize import model_from_json

import json
import logging
import os
import sys

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Occupancy Prediction - Prediction File')

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))


def getting_predictions(future_data, device_model_name):
    """
    :param future_data: dataframe
    :param device_model_name: string
    :rtype: dataframe
    """
    try:
        # load model
        with open(model_path + '/serialized_model_{0}.json'.format(device_model_name), 'r') as fin:
            m = model_from_json((json.load(fin)))
        # prediction
        predicted_data = m.predict(future_data)
        return predicted_data[['ds', 'yhat']].rename(columns={'ds': 'time', 'yhat': 'pred'})
    except Exception as e:
        logger.error(e.args)
        sys.exit(1)


def labelling_predictions(pred_data):
    """
    :param pred_data: dataframe
    :rtype: dataframe
    """
    # getting mean value
    pred_mean = pred_data['pred'].mean()
    # getting standard deviation
    pred_std = pred_data['pred'].std()
    # finding z score
    pred_data['z_score'] = pred_data['pred'].apply(lambda row: round((row - pred_mean)/pred_std, 2))
    # labelling the predictions
    pred_data['activation_predicted'] = pred_data['z_score'].apply(lambda row: 1 if row > 0 else 0)
    return pred_data[['time', 'activation_predicted']]
