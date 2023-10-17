import os
import urllib
import os.path
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import namedtuple
from utils.ACS_data_load import load_ACS_I, load_ACS_E
from utils.ACS_data_load import load_ACS
from utils.create_shift import create_shift

dirname = '/data/zhimengj/dataset'
root = 'data/'

def read_dataset(name, args, label=None, sensitive_attribute=None, fold=None):
    if args.shift=="real":
        x_train, y_train, z_train = load_ACS(dataset=name, sensitive_attributes=args.sens, 
                                            survey_year=args.ori_time, horizon= "1-Year", states = [args.ori_state])
        x_test, y_test, z_test = load_ACS(dataset=name, sensitive_attributes=args.sens, 
                                            survey_year=args.shift_time, horizon= "1-Year", states = [args.shift_state])
        return x_train, y_train, z_train[:,0], \
                x_test, y_test, z_test[:,0]
    elif args.shift=="real_iid":
        new_adult_x, new_adult_y, new_adult_s = load_ACS(dataset=name, sensitive_attributes=args.sens, 
                                            survey_year=args.ori_time, horizon= "1-Year", states = ['CA'])
        
        train_data, test_data, train_target, test_target, \
        to_protect, to_protect_test = train_test_split(new_adult_x, new_adult_y, new_adult_s, test_size=0.2)

        return train_data, train_target, to_protect[:,0], \
                test_data, test_target, to_protect_test[:,0]
    else:
        if name == 'crimes':
            y_name = label if label is not None else 'ViolentCrimesPerPop'
            z_name = sensitive_attribute if sensitive_attribute is not None else 'racepctblack'
            fold_id = fold if fold is not None else 1
            x_train, y_train, z_train, x_test, y_test, z_test = read_crimes(label=y_name, sensitive_attribute=z_name, fold=fold_id)
            
            train_ind = create_shift(np.concatenate((x_train, np.expand_dims(z_train, axis=1)), axis=1), \
                                    alpha=0.0, beta=1.0)
            test_ind = create_shift(np.concatenate((x_test, np.expand_dims(z_test, axis=1)), axis=1), \
                                    alpha=args.alpha, beta=args.beta)
            return x_train[train_ind,:], y_train[train_ind], z_train[train_ind], \
                    x_test[test_ind,:], y_test[test_ind], z_test[test_ind]

        if name=='adult':
            x_train, y_train, z_train, x_test, y_test, z_test = load_adult()

            train_ind = create_shift(np.concatenate((x_train, np.expand_dims(z_train, axis=1)), axis=1), \
                                    alpha=0.0, beta=1.0)
            test_ind = create_shift(np.concatenate((x_test, np.expand_dims(z_test, axis=1)), axis=1), \
                                    alpha=args.alpha, beta=args.beta)
            return x_train[train_ind,:], y_train[train_ind], z_train[train_ind], \
                    x_test[test_ind,:], y_test[test_ind], z_test[test_ind]
        if name=='ACS_I':
            x_train, y_train, z_train, x_test, y_test, z_test = load_ACS_I(sensitive_attributes=args.sens)

            # print(f'x_train={x_train.shape}')
            # print(f'z_train={z_train.shape}')

            train_ind = create_shift(np.concatenate((x_train, z_train), axis=1), \
                                    alpha=0.0, beta=1.0)
            test_ind = create_shift(np.concatenate((x_test, z_test), axis=1), \
                                    alpha=args.alpha, beta=args.beta)
            return x_train[train_ind,:], y_train[train_ind], z_train[train_ind, 0], \
                    x_test[test_ind,:], y_test[test_ind], z_test[test_ind, 0]
        if name=='ACS_E':
            x_train, y_train, z_train, x_test, y_test, z_test = load_ACS_E(sensitive_attributes=args.sens)
            
            train_ind = create_shift(np.concatenate((x_train, z_train), axis=1), \
                                    alpha=0.0, beta=1.0)
            test_ind = create_shift(np.concatenate((x_test, z_test), axis=1), \
                                    alpha=args.alpha, beta=args.beta)
            return x_train[train_ind,:], y_train[train_ind], z_train[train_ind, 0], \
                    x_test[test_ind,:], y_test[test_ind], z_test[test_ind, 0]
        else:
            raise NotImplemented('Dataset {} does not exists'.format(name))


def read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', fold=1):
    if not os.path.isfile(root + 'communities.data'):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", "communities.data")
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            "communities.names")

    # create names
    names = []
    with open(root + 'communities.names', 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = pd.read_csv(root + 'communities.data', names=names, na_values=['?'])

    to_drop = ['state', 'county', 'community', 'fold', 'communityname']
    data.fillna(0, inplace=True)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)

    folds = data['fold'].astype(np.int)

    y = data[label].values
    to_drop += [label]

    z = data[sensitive_attribute].values
    to_drop += [sensitive_attribute]

    data.drop(to_drop + [label], axis=1, inplace=True)

    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()

    x = np.array(data.values)
    return x[folds != fold], y[folds != fold], z[folds != fold], x[folds == fold], y[folds == fold], z[folds == fold]






#This function is a minor modification from https://github.com/jmikko/fair_ERM
def load_adult(nTrain=None, scaler=True, shuffle=False):
    if shuffle:
        print('Warning: I wont shuffle because adult has fixed test set')
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.
    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    if not os.path.isfile(root + 'adult.data'):
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data")
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", "adult.test")
    data = pd.read_csv(
        root + "adult.data",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
            )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        root + "adult.test",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"],
        skiprows=1, header=None
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    #Care there is a final dot in the class only in test set which creates 4 different classes
    target = np.array([-1.0 if (val == 0 or val==1) else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if nTrain is None:
        nTrain = len_train
    data = namedtuple('_', 'data, target')(datamat[:nTrain, :], target[:nTrain])
    data_test = namedtuple('_', 'data, target')(datamat[len_train:, :], target[len_train:])

    encoded_data = pd.DataFrame(data.data)
    encoded_data['Target'] = (data.target+1)/2

    to_protect_index = 9 ### sex
    # to_protect = 1. * (data.data[:,to_protect_index]!=data.data[:,to_protect_index][0])
    to_protect = data.data[:,to_protect_index]
    # print(f'to_protect={to_protect}')

    encoded_data_test = pd.DataFrame(data_test.data)
    encoded_data_test['Target'] = (data_test.target+1)/2
    # to_protect_test = 1. * (data_test.data[:,to_protect_index]!=data_test.data[:,to_protect_index][0])
    to_protect_test = data_test.data[:,to_protect_index]

    ### normalize age 
    norm_max = max(max(to_protect), max(to_protect_test))
    norm_min = min(min(to_protect), min(to_protect_test))

    to_protect = (to_protect - norm_min) / (norm_max - norm_min)
    to_protect_test = (to_protect_test - norm_min) / (norm_max - norm_min)

    train_data = encoded_data.drop(columns=to_protect_index)\
                            .drop('Target', axis = 1).values.astype(np.float32)
    train_target = encoded_data.drop(columns=to_protect_index)\
                            ['Target'].values.astype(np.long)
    
    test_data = encoded_data_test.drop(columns=to_protect_index)\
                            .drop('Target', axis = 1).values.astype(np.float32)
    test_target = encoded_data_test.drop(columns=to_protect_index)\
                            ['Target'].values.astype(np.long)

    #Variable to protect (9:Sex) is removed from dataset
    return train_data, train_target, to_protect, \
            test_data, test_target, to_protect_test