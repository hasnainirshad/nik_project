
GroupNormImportance
CUDA_VISIBLE_DEVICES=2 python -m svp.cifar active --proxy-arch preact56  --arch preact56 --prune-percent 0.5 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK
GroupHessianImportance
CUDA_VISIBLE_DEVICES=3 python -m svp.cifar active --proxy-arch preact56  --arch preact56 --prune-percent 0.5 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK



#Imagenet to DO 


python -m svp.imagenet active --proxy-arch resnet50 --arch resnet50 --num-workers 16 --eval-num-workers 16 --batch-size 128 --scale-learning-rates --selection-method least_confidence --prune-percent 0.5 --dataset imagenet --datasets-dir /drive1/data_HK
python -m svp.imagenet active --proxy-arch resnet50 --arch resnet50 --num-workers 16 --eval-num-workers 16 --batch-size 128 --scale-learning-rates --selection-method least_confidence --prune-percent 0.6 --dataset imagenet --datasets-dir /drive1/data_HK
python -m svp.imagenet active --proxy-arch resnet50 --arch resnet50 --num-workers 16 --eval-num-workers 16 --batch-size 128 --scale-learning-rates --selection-method least_confidence --prune-percent 0.7 --dataset imagenet --datasets-dir /drive1/data_HK


#CIfar-100 experiments

#1 -> prune 0.5
CUDA_VISIBLE_DEVICES=0 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.5 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100
#2 -> prune 0.6
CUDA_VISIBLE_DEVICES=1 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.6 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100
#3 -> prune 0.7
CUDA_VISIBLE_DEVICES=2 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.7 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100
#4 -> prune 0.8
CUDA_VISIBLE_DEVICES=3 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.8 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100



#1 -> prune 0.5 with Tsync ==1
CUDA_VISIBLE_DEVICES=0 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.5 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100  --datasets-dir /drive1/data_HK
#2 -> prune 0. 6
CUDA_VISIBLE_DEVICES=1 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.6 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100 --datasets-dir /drive1/data_HK
#3 -> prune 0.7
CUDA_VISIBLE_DEVICES=2 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.7 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100 --datasets-dir /drive1/data_HK
#4 -> prune 0.8
CUDA_VISIBLE_DEVICES=3 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.8 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100 --datasets-dir /drive1/data_HK

#4 -> prune 0.8
CUDA_VISIBLE_DEVICES=4 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.5 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10  --datasets-dir /drive1/data_HK
#2 -> prune 0. 6
CUDA_VISIBLE_DEVICES=5 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.6 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK
#3 -> prune 0.7
CUDA_VISIBLE_DEVICES=6 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.7 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK
#4 -> prune 0.8
CUDA_VISIBLE_DEVICES=7 python -m svp.cifar active --proxy-arch preact56 --prune_percent 0.8 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK


#Experiments for ICML V2 by Hu
#CIFAR-10, v2, T_sync==1
#1 -> prune 0.5 
python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.5 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10  --datasets-dir /drive1/data_HK
#2 -> prune 0. 6
python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.6 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK
#3 -> prune 0.7
python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.7 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK
#4 -> prune 0.8
python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.8 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK
----------------
#CIFAR-100, v2, T_sync==1
#1 -> prune 0.5 
CUDA_VISIBLE_DEVICES=4 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.5 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100  --datasets-dir /drive1/data_HK
#2 -> prune 0. 6
CUDA_VISIBLE_DEVICES=1 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.6 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100 --datasets-dir /drive1/data_HK
#3 -> prune 0.7
CUDA_VISIBLE_DEVICES=2 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.7 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100 --datasets-dir /drive1/data_HK
#4 -> prune 0.8
CUDA_VISIBLE_DEVICES=3 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.8 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100 --datasets-dir /drive1/data_HK


#########Trial 2#######
#CIFAR-10, v2, T_sync==1
#1 -> prune 0.5 
CUDA_VISIBLE_DEVICES=5 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.5 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10  --datasets-dir /drive1/data_HK
#2 -> prune 0. 6
CUDA_VISIBLE_DEVICES=5 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.6 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK
#3 -> prune 0.7
CUDA_VISIBLE_DEVICES=5 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.7 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK
#4 -> prune 0.8
CUDA_VISIBLE_DEVICES=6 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.8 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar10 --datasets-dir /drive1/data_HK

#CIFAR-100, v2, T_sync==1
#1 -> prune 0.5 
CUDA_VISIBLE_DEVICES=4 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.5 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100  --datasets-dir /drive1/data_HK
#2 -> prune 0. 6
CUDA_VISIBLE_DEVICES=6 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.6 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100 --datasets-dir /drive1/data_HK
#3 -> prune 0.7
CUDA_VISIBLE_DEVICES=7 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.7 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100 --datasets-dir /drive1/data_HK
#4 -> prune 0.8
CUDA_VISIBLE_DEVICES=7 python -m svp.cifar active --proxy-arch preact56 --prune-percent 0.8 --num-workers 4 --eval-num-workers 4 --selection-method least_confidence --epochs 1 --epochs 90 --epochs 45  --epochs 45 --learning-rate 0.01 --learning-rate 0.1 --learning-rate  0.01 --learning-rate  0.001 --proxy-epochs 1 --proxy-epochs 90 --proxy-epochs 45 --proxy-epochs 45 --proxy-learning-rate 0.01 --proxy-learning-rate 0.1 --proxy-learning-rate 0.01 --proxy-learning-rate 0.001 --dataset cifar100 --datasets-dir /drive1/data_HK
