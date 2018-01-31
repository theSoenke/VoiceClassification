import glob
import os
import pandas as pd
from random import shuffle


target_classes = {
    "male": 0,
    "female": 1
}


def prepare(base_path, csv, samples):
    column_name = "gender"
    colums = ['filename', column_name]
    data = pd.read_csv(os.path.join(base_path, csv), usecols=colums)
    data = data[data[column_name].notnull()]

    male_data = data[data[column_name] == "male"]
    female_data = data[data[column_name] == "female"]

    class_samples = int((samples + 1) / 2)
    male_tracks = male_data['filename'].tolist()[:class_samples]
    male_labels = male_data[column_name].tolist()[:class_samples]

    female_tracks = female_data['filename'].tolist()[:class_samples]
    female_labels = female_data[column_name].tolist()[:class_samples]

    tracks = male_tracks + female_tracks
    labels = male_labels + female_labels
    shuffle(tracks)
    shuffle(labels)

    return tracks, labels, target_classes
