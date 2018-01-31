import glob
import os
import pandas as pd
import sys

target_classes = {
    'singapore': 0,
    'scotland': 1,
    'indian': 2,
    'ireland': 3,
    'african': 4,
    'philippines': 5,
    'england': 6,
    'southatlandtic': 7,
    'wales': 8,
    'malaysia': 9,
    'hongkong': 10,
    'newzealand': 11,
    'australia': 12,
    'us': 13,
    'bermuda': 14,
    'canada': 15
}


def prepare(base_path, csv, samples):
    column_name = "accent"
    colums = ['filename', column_name]
    data = pd.read_csv(os.path.join(base_path, csv), usecols=colums)
    data = data[data[column_name].notnull()]
    tracks = data['filename'].tolist()[:samples]
    labels = data[column_name].tolist()[:samples]

    return tracks, labels, target_classes
