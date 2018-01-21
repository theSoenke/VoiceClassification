import argparse
import concurrent.futures
import librosa
import numpy as np
import os
import tensorflow as tf
import gender_data as gender_data


def get_track_features(path, time_series_length, features_size, hop_length, index):
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    features = np.zeros((time_series_length, features_size))
    features[:, 0:13] = mfcc.T[0:time_series_length, :]
    features[:, 13:14] = spectral_center.T[0:time_series_length, :]
    features[:, 14:26] = chroma.T[0:time_series_length, :]
    features[:, 26:33] = spectral_contrast.T[0:time_series_length, :]
    return (features, index)


def extract(base_path, track_paths, labels, target_classes, time_series_length, features_size, hop_length):
    data = np.zeros((len(track_paths), time_series_length, features_size))
    classes = []
    futures = []
    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        for i, track in enumerate(track_paths):
            classes.append(target_classes[labels[i]])
            print("Extracting ", track)
            future = executor.submit(get_track_features, os.path.join(base_path, track), time_series_length, features_size, hop_length, i)
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            data[result[1]] = result[0]

    data = tf.keras.utils.normalize(data)
    return data, np.array(classes)


def save(features, classes, classifier, set):
    save_features_name = classifier + "features-" + set + ".npy"
    save_classes_name = classifier + "classes-" + set + ".npy"
    with open(save_features_name, "wb") as f:
        np.save(f, features)
    with open(save_classes_name, "wb") as f:
        np.save(f, classes)


def extract_gender(base_path, time_series_length, features_size, hop_length):
    print("Prepare train set")
    tracks, labels, target_classes = gender_data.prepare(base_path, "cv-valid-train.csv", 500)
    features, labels = extract(base_path, tracks, labels, target_classes, time_series_length, features_size, hop_length)
    save(features, labels, "gender", "train")

    print("Prepare test set")
    tracks, labels, target_classes = gender_data.prepare(base_path, "cv-valid-test.csv", 50)
    features, labels = extract(base_path, tracks, labels, target_classes, time_series_length, features_size, hop_length)
    save(features, labels, "gender", "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_series_length", type=int, default=128)
    parser.add_argument("--data")
    FLAGS, unknown = parser.parse_known_args()
    base_path = FLAGS.data
    time_series_length = FLAGS.time_series_length
    hop_length = 128
    features_size = 33

    extract_gender(base_path, time_series_length, features_size, hop_length)


if __name__ == "__main__":
   main()
