#!/bin/bash

zip archive.zip main.py model.py util.py attention.py randomForest.py gradient_boosting.py

scp -i ~/.ssh/vegamissile archive.zip victor@vega.mont.priv:/home/victor/tsf
