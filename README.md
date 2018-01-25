# VoiceClassification

Voice classification based on the [Common Voice](https://voice.mozilla.org/data) dataset

## Install
1. `python3 -m pip install --user pipenv`
2. `pipenv shell`
3. `pipenv install`

## Train
1. `pipenv shell`
2. `python3 extract_features.py --data ~/Downloads/cv_corpus_v1`
3. `python3 train.py --steps 100 --samples 1000`

## Cluster
1. Submit job `sbatch slurm_train.sh`