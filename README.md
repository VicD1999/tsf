# TSF

This repository contains the software supporting the results of my master 
thesis about wind power forecasting which is available here: 
https://matheo.uliege.be/handle/2268.2/14587

# Install packages

Use pip to install all the packages required to run this software:
```
pip install -r requirements.txt
```

# Repository structure
```
.
├── GEFcom2014.ipynb
├── README.md
├── bash
│   ├── export.bash
│   ├── get_results.bash
│   └── import.bash
├── data
│   └── output15
│       ├── dataset0_15.csv
│       ├── dataset1_15.csv
│       └── dataset2_15.csv
├── python_code
│   ├── attention.py
│   ├── comparison.py
│   ├── concat.py
│   ├── dataset.py
│   ├── gefcom.py
│   ├── main.py
│   ├── model.py
│   ├── naive.py
│   ├── randomForest.py
│   ├── transformer.py
│   └── util.py
└── requirements.txt
```

# Datasets

This repository uses two datasets:
    1) A concatenation of MAR data and ORES data
    2) Gefcom Dataset available here: http://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html

# How to use the software

First build the datasets for sklearn and torch models using ```dataset.py```.
Then, you can use for example the following commands:

Sklearn:
```python python_code/randomForest.py -g -t --model RandomForestRegressor --farm 1```

Torch models:
```python python_code/main.py -g -t --rnn simple_rnn --farm 1 ```

The parser will give you the explainations for each arguments.

# Cite

If you use this repository please consider citing :
    Dachet, V. (2022). Master thesis : Wind Power Forecasting. (Unpublished master's thesis). Université de Liège, Liège, Belgique. Retrieved from https://matheo.uliege.be/handle/2268.2/14587
