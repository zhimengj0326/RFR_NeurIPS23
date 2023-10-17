#!/bin/bash

# hyper_pent = 0. 0.1 0.3 0.7 1.0 3.0 5.0 10.0 20.0 50.0 100.0 500.0
# datasets = 'crimes' 'adult' 'ACS_I' 'ACS_E'
# alpha, beta = (0.0, 1.0), (1.0, 2.0), (1.5, 3.0), (3.0, 6.0), (5.0, 10.0), (7.0, 10.5)

for dataset in 'adult' ## 'ACS_I' 'ACS_E'
do
    for hyper_pent in 0.0 0.1 0.5 1.0 3.0 5.0 10.0 30.0 50.0 100.0 200.0 ## 0.0 0.1 0.5 1.0 3.0 5.0 10.0 30.0 50.0 100.0 200.0
    do
        python -u main_dnn.py --data $dataset --times 5 --n_epoch 200 --batch_size 200 --gpu 0  \
                --alpha 1.5 --beta 3.0 --hyper_pent $hyper_pent &
        python -u main_dnn.py --data $dataset --times 5 --n_epoch 200 --batch_size 200 --gpu 3  \
                --alpha 1.0 --beta 2.0 --hyper_pent $hyper_pent &
        python -u main_dnn.py --data $dataset --times 5 --n_epoch 200 --batch_size 200 --gpu 2  \
                --alpha 3.0 --beta 6.0 --hyper_pent $hyper_pent &
    done
    wait
done
# python -u main_dnn.py --data 'adult' --times 1 --n_epoch 10 --batch_size 200 --gpu 1  \
#                             --alpha 0.0 --beta 1.0 --hyper_pent 0.1