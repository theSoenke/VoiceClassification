import glob
import os
import pandas as pd


target_classes = {
    "male": 0,
    "female": 1
}


def prepare(base_path, csv_name, num_samples):
    colums = ['filename', 'gender']
    data_gender = pd.read_csv(os.path.join(base_path, csv_name), usecols=colums)
    data_gender = data_gender[data_gender["gender"].notnull()]
    data_gender = data_gender[data_gender["gender"] != "other"]
    tracks = data_gender['filename'].tolist()[:num_samples]
    labels = data_gender['gender'].tolist()[:num_samples]

    return tracks, labels, target_classes
