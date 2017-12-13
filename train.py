import os
import csv
import pandas as pd
import numpy as np
from numpy import genfromtxt
import tensorflow as tf

np.random.seed(42)


def main():
    base_path = os.environ['DATASET']
    train_path = os.path.join(base_path, "cv-valid-train.csv")
    train_labels = pd.read_csv(train_path)
    print(train_labels)


if __name__ == "__main__":
    main()
