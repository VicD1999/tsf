#!/bin/bash

zip archive.zip python_code/*.py

scp -i ~/.ssh/vegamissile archive.zip victor@vega.mont.priv:/home/victor/tsf
