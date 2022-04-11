#!/bin/bash

echo "Fisrt agument: name of the model"
echo "Second agument: number of the model"

scp -i ~/.ssh/vegamissile victor@vega.mont.priv:/home/victor/tsf/results/*.csv results

scp -i ~/.ssh/vegamissile victor@vega.mont.priv:/home/victor/tsf/model/$1/*.model model/$1

