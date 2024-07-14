
#!/bin/bash

CURDIR=/home/mila/y/yuyan.chen/projects/anuraset

sbatch $CURDIR/train.sh --config=$CURDIR/baseline/configs/exp_resnet18.yaml 

sbatch $CURDIR/train.sh --config=$CURDIR/baseline/configs/exp_resnet50.yaml 

sbatch $CURDIR/train.sh --config=$CURDIR/baseline/configs/exp_resnet152.yaml 
