#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import timedelta

import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Occupancy Prediction - Util Functions')


def grouping_device_data(df, time_frame):
    """
    :param df: dataframe
    :param time_frame: int
    :rtype: dataframe
    """
    # Interval will be written as parametric for robust forecast - 2min, 5min, 15min, 30min, 60min (requested & default)
    return df.groupby(pd.Grouper(key='time', freq='%smin' % time_frame)).sum().reset_index().rename(
        columns={'time': 'ds', 'device_activated': 'y'})  # y: number of occupancy or event count


def creating_date_range(start_date, end_date, time_frame):
    """
    :param start_date: datetime
    :param end_date: datetime
    :param time_frame: int
    :return: datetime
    """
    delta = timedelta(minutes=time_frame)
    while start_date < end_date:
        yield start_date
        start_date += delta


def creating_dummy_date_data(starting_date, finishing_date, time_frame):
    """
    :param starting_date: datetime
    :param finishing_date: datetime
    :param time_frame: int
    :rtype: dataframe
    """
    dates = []
    dates_df = pd.DataFrame(columns=['ds'])

    logger.info("Creating dummy date data...")
    for single_date in creating_date_range(start_date=starting_date, end_date=finishing_date, time_frame=time_frame):
        dates.append(single_date.strftime('%Y-%m-%d %H:%M:%S'))

    dates_df['ds'] = dates
    dates_df['ds'] = pd.to_datetime(dates_df['ds'])
    return dates_df

