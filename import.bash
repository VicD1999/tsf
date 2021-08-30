#!/bin/bash

echo "Fisrt agument: name of the model"
echo "Second agument: number of the model"

scp -i ~/.ssh/vega victor@vega.montefiore.ulg.ac.be:/home/victor/tsf/results/*.csv results

scp -i ~/.ssh/vega victor@vega.montefiore.ulg.ac.be:/home/victor/tsf/model/$1/*.model model/$1

