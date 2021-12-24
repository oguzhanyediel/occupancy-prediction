#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime, time, timedelta

import argparse
import config
import logging
import model
import os
import pandas as pd
import prediction
import utils

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Occupancy Prediction - Main File')

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


def arg_parser():
    """
    :return: parser argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--timestamp', help='Starting date of prediction, default is end date of the raw data',
                        required=False)
    parser.add_argument('-IF', '--input_file_csv', help='Input file name', required=True)
    parser.add_argument('-OF', '--output_file_csv', help='Output file name', required=True)
    parser.add_argument('-TF', '--time_frame', help='Forecast data interval in terms of minutes; 2,5,15,30,60,120; '
                                                    'default 60 (requested)', required=False, default=60)
    return parser.parse_args()


def getting_raw_data(input_file_name):
    """
    :param input_file_name: string
    :rtype: dataframe
    """
    logger.info("Reading csv file...")
    raw_data = pd.read_csv(data_path + '/' + input_file_name)

    logger.info("Creating str date to timestamp...")
    raw_data['time'] = pd.to_datetime(raw_data['time'])
    return raw_data


def creating_model_data(raw_df, device, time_frame):
    """
    :param raw_df: dataframe
    :param device: string
    :param time_frame: int
    :rtype: dataframe
    """
    logger.info("Creating starting and finishing date of dummy date data...")
    starting_date = datetime.combine(raw_df['time'].min(), datetime.min.time())
    finishing_date = datetime.combine(raw_df['time'].max(), time(0, 0)) + timedelta(1)

    logger.info("Getting single device data...")
    selected_device_data = raw_df[raw_df['device'] == device][['time', 'device_activated']].reset_index(drop=True)

    logger.info("Creating event count on basis of given time interval...")
    grouping_data = utils.grouping_device_data(selected_device_data, time_frame)

    logger.info("Getting dataframe that contains dummy date for robust forecast...")
    dummy_date_data = utils.creating_dummy_date_data(starting_date, finishing_date, time_frame)

    logger.info("Creating processed model data...")
    return pd.merge(dummy_date_data, grouping_data, on='ds', how='left').fillna(0)


def creating_future_data(raw_df, timestamp_, time_frame):
    """
    :param raw_df: dataframe
    :param timestamp_: string
    :param time_frame: int
    :rtype: dataframe
    """
    finishing_date = datetime.combine(raw_df['time'].max(), time(0, 0)) + timedelta(1)

    if timestamp_ is None:
        fc_starting_date = finishing_date
        fc_finishing_date = finishing_date + timedelta(days=1)
    else:
        fc_starting_date = datetime.combine(datetime.strptime(timestamp_, '%Y-%m-%d %H:%M:%S'),
                                            time(0, 0)) + timedelta(1)
        fc_finishing_date = fc_starting_date + timedelta(days=1)

    return utils.creating_dummy_date_data(starting_date=fc_starting_date, finishing_date=fc_finishing_date,
                                          time_frame=time_frame)


def app(timestamp_, time_frame):
    """
    :param timestamp_: string
    :param time_frame: int
    :rtype: dataframe
    """
    submission_data = pd.DataFrame(columns=['time', 'device', 'activation_predicted'])

    for i in range(1, config.device_count + 1):
        device_number = 'device_%s' % i

        # getting processed model data
        model_data = creating_model_data(raw_df=rdata, device=device_number, time_frame=time_frame)

        # dropping duplicate records
        unique_model_data = model_data.drop_duplicates()

        # preparing data to be forecasted
        future_data = creating_future_data(raw_df=rdata, timestamp_=timestamp_, time_frame=time_frame)

        # modelling data
        model.fit_predict_model(model_data=unique_model_data, device_model_name=config.dmn[device_number])

        # predicting new data
        pred_data = prediction.getting_predictions(future_data=future_data, device_model_name=config.dmn[device_number])

        # labelling the predictions
        result = prediction.labelling_predictions(pred_data=pred_data)

        # adding device
        result['device'] = device_number

        # adding the submission dataframe
        submission_data = submission_data.append(result, ignore_index=True)

    return submission_data.sort_values(['time', 'device'])  # for requested file content, not necessary normally


if __name__ == "__main__":
    args = arg_parser()
    rdata = getting_raw_data(args.input_file_csv)

    # operating the application
    submission_df = app(args.timestamp, int(args.time_frame))

    # Yay!
    submission_df.to_csv(data_path + '/' + args.output_file_csv, index=False)
