# common-voice

## Install
1. `python3 -m pip install --user virtualenv`
2. `python3 -m virtualenv env`
3. `source env/bin/activate`
4. `pip install -r requirements.txt`

## Run
1. Extract data `python3 data_extraction.py --data ~/Downloads/cv_corpus_v1`
2. Train `python3 train.py`