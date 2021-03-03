#!/bin/bash

mkdir data
python ./src/get_data.py
unzip ./data/data.zip -d ./data