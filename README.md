# How to organize a model training repository - a tutorial

Stage 1: 

Split the initial notebooks into `/src/model.py`, `/src/dataset.py `, `/src/utils.py` and `train.py` in the following basic structure.

├── src
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
└── train.py

Add `requirements.txt` for better installation.