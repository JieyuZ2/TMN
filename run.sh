#!/bin/bash

cd $(dirname $0)

mkdir data
hadoop fs -copyToLocal hdfs://haruna/user/lab/zhangjieyu/TaxoExpan/data/MAG-PSY ./data
hadoop fs -copyToLocal hdfs://haruna/user/lab/zhangjieyu/TaxoExpan/data/MAG-CS ./data
hadoop fs -copyToLocal hdfs://haruna/user/lab/zhangjieyu/TaxoExpan/data/SemEval-Noun ./data
hadoop fs -copyToLocal hdfs://haruna/user/lab/zhangjieyu/TaxoExpan/data/SemEval-Verb ./data
rm -rf ./config_files
hadoop fs -copyToLocal hdfs://haruna/user/lab/zhangjieyu/TaxoExpan/config_files ./
python3 train.py $@
echo 1
hadoop fs -copyFromLocal -f ./data/saved hdfs://haruna/user/lab/zhangjieyu/TaxoExpan/data
