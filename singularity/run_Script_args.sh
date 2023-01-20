#!/bin/bash

 

echo $2

 

source /opt/conda/bin/activate SCG_OL
#conda activate SCG_OL
cd /app
#pip install -e .
python $1 $2

 


#homogenous/train_E2E_batch_4layers.py