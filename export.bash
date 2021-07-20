#!/bin/bash

zip archive.zip main.py model.py util.py

scp -i ~/.ssh/vega archive.zip victor@vega.montefiore.ulg.ac.be:/home/victor/tsf