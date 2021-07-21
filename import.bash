#!/bin/bash

echo "Fisrt agument: name of the model"
echo "Second agument: number of the model"

scp -i ~/.ssh/vega victor@vega.montefiore.ulg.ac.be:/home/victor/tsf/results/*.csv /Users/victordachet/Documents/Projet_Ernst_Fettweis/2021/tsf/results

scp -i ~/.ssh/vega victor@vega.montefiore.ulg.ac.be:/home/victor/tsf/model/$1/$2.model /Users/victordachet/Documents/Projet_Ernst_Fettweis/2021/tsf/model/$1

