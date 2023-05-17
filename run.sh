#!/usr/bin/env bash
# create virtual environment called lang_modelling_env
python3 -m venv indofashion_env

# activate virtual environment
source ./indofashion_env/bin/activate

# install requirements
python3 -m pip install -r requirements.txt

# run script for training and classifying
python3 src/main.py

# deactivate virtual environment
deactivate indofashion_env