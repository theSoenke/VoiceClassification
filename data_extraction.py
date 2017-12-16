import os
import numpy as np
import glob
import librosa
import argparse
import pandas as pd
import concurrent.futures

target_dict = {
    "male": 0,
    "female": 1
}


def extract_track_feature(path, index):
    print("Extracting ", path)
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH)

    feature_matrix = np.zeros((TIME_SERIES_LENGTH, MERGED_FEATURES_SIZE))
    feature_matrix[:, 0:13] = mfcc.T[0:TIME_SERIES_LENGTH, :]
    feature_matrix[:, 13:14] = spectral_center.T[0:TIME_SERIES_LENGTH, :]
    feature_matrix[:, 14:26] = chroma.T[0:TIME_SERIES_LENGTH, :]
    feature_matrix[:, 26:33] = spectral_contrast.T[0:TIME_SERIES_LENGTH, :]
    return (feature_matrix, index)


def extract_features(base_path, track_paths, gender):
    data = np.zeros(
        (len(track_paths), TIME_SERIES_LENGTH, MERGED_FEATURES_SIZE))
    classes = []
    futures = []
    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        for i, track in enumerate(track_paths):
            classes.append(target_dict[gender[i]])
            future = executor.submit(extract_track_feature, os.path.join(base_path, track), i)
    for future in concurrent.futures.as_completed(futures):
        data[future[1]] = future[0]

    return data, np.array(classes)


def save_features(features, classes, name):
    save_features_name = "features-" + name + ".npy"
    save_classes_name = "classes-" + name + ".npy"
    with open(save_features_name, "wb") as f:
        np.save(f, features)
    with open(save_classes_name, "wb") as f:
        np.save(f, classes)


def prepare_data():
    colums = ['filename', 'gender']
    train_data_gender = pd.read_csv(os.path.join(
        DATA_PATH, "cv-valid-train.csv"), usecols=colums)
    train_data_gender = train_data_gender[train_data_gender["gender"].notnull()]
    train_data_gender = train_data_gender[train_data_gender["gender"] != "other"]

    train_tracks = train_data_gender['filename'].tolist()
    train_labels = train_data_gender['gender'].tolist()
    train_features, train_labels = extract_features(
        DATA_PATH,
        train_tracks,
        train_labels)
    save_features(train_features, train_labels, "train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_series_length", type=int, default=128)
    parser.add_argument("--data")
    FLAGS, unknown = parser.parse_known_args()
    HOP_LENGTH = 128
    TIME_SERIES_LENGTH = FLAGS.time_series_length
    DATA_PATH = FLAGS.data
    MERGED_FEATURES_SIZE = 33

    prepare_data()
