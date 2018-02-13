import glob
import os
import pandas as pd


target_classes = {
    "male": 0,
    "female": 1
}


def prepare(base_path, csv, samples):
    column_name = "gender"
    colums = ['filename', column_name]
    data = pd.read_csv(os.path.join(base_path, csv), usecols=colums)
    data = data[data[column_name].notnull()]
    data = data[data[column_name] != "other"]
    tracks = data['filename'].tolist()[:samples]
    labels = data[column_name].tolist()[:samples]

    return tracks, labels, target_classes
