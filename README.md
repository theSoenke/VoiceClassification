# VoiceClassification

Voice classification based on the [Common Voice](https://voice.mozilla.org/data) dataset

## Install
1. `python3 -m pip install --user pipenv`
2. `pipenv shell`
3. `pipenv install`

## Train
1. Run `pipenv shell`
2. Extract data `python3 data_extraction.py --data ~/Downloads/cv_corpus_v1`
3. Train `python3 train.py`