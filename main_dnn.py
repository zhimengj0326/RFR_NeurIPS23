import sys, os
# sys.path.append(os.path.abspath(os.path.join('../..')))

import torch
import numpy as np
import torch.utils.data as data_utils
from torch import nn
import torch.nn.functional as F
from utils.data_load import read_dataset

from sklearn.metrics import accuracy_score,roc_auc_score,recall_score
from sklearn.metrics import average_precision_score
import sys
import random
import argparse
import logging
import time

from models import NetRegression, NeuralGBDT

from utils.fair_metric import dp, eo

class NetRegression(nn.Module):
    def __init__(self, input_size, num_classes, size = 50):
        super(NetRegression, self).__init__()
        self.first = nn.Linear(input_size, size)
        self.fc = nn.Linear(size, size)
        self.last = nn.Linear(size, num_classes)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        out = F.selu(self.first(x))
        out = F.selu(self.fc(out))
        out = self.last(out)
        return self.sigmoid(out)


def regularized_learning(dataset_loader, x_train, y_train, z_train, x_test, y_test, z_test, \
                        model, device_gpu, logger, penalty_coefficient, \
                        data_fitting_loss, lr=1e-5, num_epochs=10):

    # mse regression objective
    # data_fitting_loss = nn.MSELoss()

    # stochastic optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    for j in range(num_epochs):
        for i, (x, y, z) in enumerate(dataset_loader):
            optimizer.zero_grad()
            outputs = model(x).flatten()
            # print(f'outputs={outputs}')
            # print(f'y={y}')
            loss = data_fitting_loss(outputs, y)
            # loss += penalty_coefficient * fairness_penalty(outputs, z, device_gpu)
            loss += penalty_coefficient * dp(outputs, z)

            loss.backward()
            optimizer.step()
        if dataset_name == 'crimes':
            loss_train, loss_test, mae_train, mae_test, DP_train, DP_test = evaluate(model, data_fitting_loss, \
                                    x_train, y_train, z_train, x_test, y_test, z_test)
            logger.info('epoch: {}:'.format(j))
            logger.info(f'train loss: {loss_train:.4f}, train mae: {mae_train:.4f}, DP train: {DP_train:.4f}')
            logger.info(f'test loss: {loss_test:.4f}, test mae: {mae_test:.4f}, DP test: {DP_test:.4f}')
        else:
            loss_train, loss_test, acc_train, acc_test, DP_train, DP_test = evaluate(model, data_fitting_loss, \
                                    x_train, y_train, z_train, x_test, y_test, z_test)
            logger.info('epoch: {}:'.format(j))
            logger.info(f'train loss: {loss_train:.4f}, train acc: {acc_train:.4f}, DP train: {DP_train:.4f}')
            logger.info(f'test loss: {loss_test:.4f}, test acc: {acc_test:.4f}, DP test: {DP_test:.4f}')

def evaluate_reg(model, data_fitting_loss, x_train, y_train, z_train, x, y, z):
    prediction = model(x_train).detach().flatten()
    loss_train = data_fitting_loss(prediction, y_train).item()
    mae_train = nn.L1Loss()(prediction, y_train).item()
    # DP_train = fairness_penalty(prediction, z_train, device_gpu).item()
    DP_train = dp(prediction, z_train).item()


    prediction = model(x).detach().flatten()
    loss_test = data_fitting_loss(prediction, y).item()
    mae_test = nn.L1Loss()(prediction, y_test).item()
    # DP_test = fairness_penalty(prediction, z, device_gpu).item()
    DP_test = dp(prediction, z).item()
    return loss_train, loss_test, mae_train, mae_test, DP_train, DP_test

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0.5).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct.item() / len(labels)
    return acc

def evaluate_class(model, data_fitting_loss, x_train, y_train, z_train, x, y, z):
    prediction = model(x_train).detach().flatten()
    loss_train = data_fitting_loss(prediction, y_train).item()
    # DP_train = fairness_penalty(prediction, z_train, device_gpu).item()
    DP_train = dp(prediction, z_train).item()

    acc_train = accuracy(prediction, y_train)

    prediction = model(x).detach().flatten()
    loss_test = data_fitting_loss(prediction, y).item()
    acc_test = accuracy(prediction, y_test)
    # DP_test = fairness_penalty(prediction, z, device_gpu).item()
    DP_test = dp(prediction, z).item()
    return loss_train, loss_test, acc_train, acc_test, DP_train, DP_test

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('--data', type=str, help='data sources to use', default='crimes')
parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--times', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--check_sol', type=float, default=1e-3, help='check solution')
parser.add_argument('--hyper_pent', type=float, default=1.0, help='Hyperparmeters for penalty')
parser.add_argument('--alpha', type=float, default=0.0, help='Prior Gaussian distribution - mean pert')
parser.add_argument('--beta', type=float, default=1.0, help='Prior Gaussian distribution - varaince amp')

parser.add_argument('--shift', type=str, choices=['syn', 'real', 'real_iid'], default='syn', help='Distribution shift type')
parser.add_argument('--real_shift', type=str, choices=['states', 'time'], default='states', help='Real distribution shift type')
parser.add_argument('--ori_state', type=str, choices=['CA', 'MI'], default='CA', help='Shift across state')
parser.add_argument('--ori_time', type=str, choices=['2018', '2016', '2015'], default='2018', help='Shift across times')
parser.add_argument('--shift_state', type=str, choices=['CA', 'MI'], default='CA', help='Shift across state')
parser.add_argument('--shift_time', type=str, choices=['2018', '2016', '2015'], default='2018', help='Shift across times')
parser.add_argument('--sens', type=str, default='sex', help='Sensitive Attribute')
parser.add_argument('--model_save', type=str, default='False', help='save model checkpoint')
parser.add_argument('--model', type=str, default='MLP', help='model backbone')
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

batch_size = args.batch_size
GPU = args.gpu
num_epochs = args.n_epoch
lr = args.lr
test_sol = args.check_sol
penalty_coefficient = args.hyper_pent
dataset_name = args.data
RUNNING_TIME = args.times

x_train, y_train, z_train, x_test, y_test, z_test = read_dataset(name=dataset_name, args=args, fold=1)
n, d = x_train.shape
print(f'x_train size = {x_train.shape}')
print(f'x_test size = {x_test.shape}')
print(f'x_all size = {x_train.shape[0] + x_test.shape[0]}')


device_gpu = torch.device('cuda:{}'.format(GPU))

if dataset_name == 'crimes':
    data_fitting_loss = nn.MSELoss()
    evaluate = evaluate_reg
else:
    data_fitting_loss = nn.functional.binary_cross_entropy  ##nn.CrossEntropyLoss()
    evaluate = evaluate_class

# num_epochs = 20
# lr = 1e-5

# penalty_coefficient = 1.0

# wrap dataset in torch tensors
# print(f'y_train={y_train}')
y_train = torch.tensor(y_train.astype(np.float32)).to(device_gpu)
x_train = torch.tensor(x_train.astype(np.float32)).to(device_gpu)
z_train = torch.tensor(z_train.astype(np.float32)).to(device_gpu)
z_train_bn = (z_train>0.5).float()

  
dataset = data_utils.TensorDataset(x_train, y_train, z_train)

dataset_loader = data_utils.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# print(f'x_test={type(x_test)}')

y_test = torch.tensor(y_test.astype(np.float32)).to(device_gpu)
x_test = torch.tensor(x_test.astype(np.float32)).to(device_gpu)
z_test = torch.tensor(z_test.astype(np.float32)).to(device_gpu)

performances = []
fairnesss = []
for run_time in range(RUNNING_TIME):
    t_total = time.time()
    # run_time = 0
    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    if args.model == "MLP":
        parent_folder = 'log/'
    else:
        parent_folder = 'log/GBDT/'

    if args.shift == "syn":  ## synthetic distribution shift
        log_path = parent_folder + f'{dataset_name}/{args.shift}/reg/alpha={args.alpha}_beta={args.beta}'
    elif args.shift == "real_iid": ## iid
        log_path = parent_folder + f'{dataset_name}/{args.shift}/reg'
    else:
        if args.real_shift=='states':    ## distribution shift across state
            log_path = parent_folder + f'{dataset_name}/{args.shift}/reg/{args.ori_state}-{args.shift_state}'
        else:    ## distribution shift across time
            log_path = parent_folder + f'{dataset_name}/{args.shift}/reg/{args.ori_time}-{args.shift_time}'

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    fh = logging.FileHandler(log_path + f'/{args.prefix}-hyper={penalty_coefficient}-{run_time}.log', mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    if args.model == "MLP":
        model = NetRegression(d, 1).to(device_gpu)
    else:
        model = NeuralGBDT(input_dim=d, output_dim=1, n_ensemble=4).to(device_gpu)

    regularized_learning(dataset_loader, x_train, y_train, z_train, x_test, y_test, z_test, \
                                model, device_gpu, logger, penalty_coefficient, \
                                data_fitting_loss, lr=lr, num_epochs=num_epochs)
    if dataset_name == 'crimes':
        loss_train, loss_test, mae_train, mae_test, DP_train, DP_test = evaluate(model, data_fitting_loss, \
                                    x_train, y_train, z_train, x_test, y_test, z_test)
        print(f'loss train: {loss_train:.4f}, mae train: {mae_train:.4f}, DP train: {DP_train:.4f}')
        print(f'loss test: {loss_test:.4f}, mae test: {mae_test:.4f}, DP test: {DP_test:.4f}')
        ## record performance and fairness metrics
        performances.append([loss_train, loss_test, mae_train, mae_test])
        fairnesss.append([DP_train, DP_test])
    else:
        loss_train, loss_test, acc_train, acc_test, DP_train, DP_test = evaluate(model, data_fitting_loss, \
                                    x_train, y_train, z_train, x_test, y_test, z_test)
        print(f'loss train: {loss_train:.4f}, acc train: {acc_train:.4f}, DP train: {DP_train:.4f}')
        print(f'loss test: {loss_test:.4f}, acc test: {acc_test:.4f}, DP test: {DP_test:.4f}')
        ## record performance and fairness metrics
        performances.append([loss_train, loss_test, acc_train, acc_test])
        fairnesss.append([DP_train, DP_test])

    print(f'running time={time.time() - t_total}')
    if run_time < RUNNING_TIME - 1:
        fh.close()
        logger.removeHandler(fh)

if args.model_save=='True':
    checkpoints_path = log_path + f'/{args.prefix}-hyper={penalty_coefficient}.pt'
    torch.save(model, checkpoints_path)

### statistical results
performance_mean = np.around(np.mean(performances, 0), 4)
performance_std = np.around(np.std(performances, 0), 4)
fairness_mean = np.around(np.mean(fairnesss, 0), 4)
fairness_std = np.around(np.std(fairnesss, 0), 4)

logger.info('Average of performance and fairness metric')
logger.info("Test statistics: -- train loss: {:.4f}+-{:.4f} , test loss: {:.4f}+-{:.4f}" \
            .format(performance_mean[0], performance_std[0], 
                    performance_mean[1], performance_std[1]))
if dataset_name == 'crimes':
    logger.info("Test statistics: -- train mae: {:.4f}+-{:.4f} , test mae: {:.4f}+-{:.4f}" \
                .format(performance_mean[2], performance_std[2], 
                        performance_mean[3], performance_std[3]))
else:
    logger.info("Test statistics: -- train acc: {:.4f}+-{:.4f} , test acc: {:.4f}+-{:.4f}" \
                .format(performance_mean[2], performance_std[2], 
                        performance_mean[3], performance_std[3]))

logger.info('Test statistics: -- train D_SP: {:.4f}+-{:.4f}, test D_SP: {:.4f}+-{:.4f}'\
            .format(fairness_mean[0], fairness_std[0], 
                    fairness_mean[1], fairness_std[1]))
fh.close()
logger.removeHandler(fh)