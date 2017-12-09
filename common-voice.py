import os
import tensorflow as tf
import numpy as np


def preprocessing(path):
    print(path)


def main():
    path = os.environ['DATASET']
    preprocessing(path)


if __name__ == "__main__":
    main()