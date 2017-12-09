import os
import csv
import tensorflow as tf
import numpy as np
from numpy import genfromtxt


def read_csv(path):
    data = genfromtxt(path, delimiter=',', skip_header=1, dtype=None)
    return data


def main():
    base_path = os.environ['DATASET']
    train_path = os.path.join(base_path, "cv-valid-train.csv")
    train_labels = read_csv(train_path)
    print(train_labels)


if __name__ == "__main__":
    main()
