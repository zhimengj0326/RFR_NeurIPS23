#!/bin/bash

# alpha_values = 0.0 10.0 100.0 500.0 1000.0 1500.0 2000.0 3000.0
# datasets = 'crimes' 'adult' 'ACS_I' 'ACS_E'
for dataset in 'ACS_E'
do
    for alpha_value in 0.0 10.0 100.0 500.0 1000.0 1500.0 2000.0 3000.0
    do
        # python -u main_dnn_adv.py --data $dataset --times 5 --n_epoch 200 --batch_size 200 --gpu 2  \
        #             --alpha 0.0 --beta 1.0 --hyper_pent $alpha_value &
        # python -u main_dnn_adv.py --data $dataset --times 5 --n_epoch 200 --batch_size 200 --gpu 3  \
        #             --alpha 1.0 --beta 2.0 --hyper_pent $alpha_value &
        # python -u main_dnn_adv.py --data $dataset --times 5 --n_epoch 200 --batch_size 200 --gpu 3  \
        #             --alpha 1.5 --beta 3.0 --hyper_pent $alpha_value &
        python -u main_dnn_adv.py --data $dataset --times 5 --n_epoch 200 --batch_size 200 --gpu 0  \
                    --alpha 3.0 --beta 6.0 --hyper_pent $alpha_value &
    done
    wait
done