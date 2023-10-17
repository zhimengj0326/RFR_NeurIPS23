#!/bin/bash

# hyper_pent = 0. 0.1 0.3 0.7 1.0 3.0 5.0 10.0 20.0 50.0 100.0 500.0
# datasets = 'crimes' 'adult' 'ACS_I' 'ACS_E'

for dataset in 'ACS_I' ## 'ACS_I' 'ACS_E'
do
    for hyper_pent in 0.0 0.1 0.5 1.0 3.0 5.0 ## hyper_pent in 0.0 0.1 0.5 1.0 3.0 10.0 30.0 50.0 100.0 200.0
    do
        ## across state
        python -u main_rfr.py --data $dataset --times 1 --n_epoch 200 --batch_size 200 --gpu 0  --ablation True --model_save True \
        --shift 'real' --real_shift 'states' --ori_state 'MI' --shift_state 'CA' --hyper_pent $hyper_pent &

        ## across time
        python -u main_rfr.py --data $dataset --times 1 --n_epoch 200 --batch_size 200 --gpu 1  --ablation True --model_save True \
        --shift 'real' --real_shift 'time' --ori_time '2016' --shift_time '2018' --hyper_pent $hyper_pent &
        
    done
    wait

done