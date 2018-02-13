import glob
import os
import pandas as pd
import numpy as np


target_classes = {
    "teens": 0,
    "twenties": 1,
    "thirties": 2,
    "fourties": 3,
    "fifties": 4,
    "sixties": 5,
    "seventies": 6
}


def prepare(base_path, csv, samples):
    column_name = 'age'
    colums = ['filename', column_name]
    data = pd.read_csv(os.path.join(base_path, csv), usecols=colums)
    data = data[data[column_name].notnull()]
    data = data[data[column_name] != 'eighties'] # not enough samples
    data = data[data[column_name] != 'nineties']
    tracks = data['filename'].tolist()[:samples]
    labels = data[column_name].tolist()[:samples]

    return tracks, labels, target_classes
