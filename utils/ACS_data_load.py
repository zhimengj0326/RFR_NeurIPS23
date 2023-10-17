import folktables
from folktables import ACSDataSource, ACSEmployment, ACSIncome
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data_root = '/data/zhimengj/dataset'

def load_folktables_income(sensitive_attributes="sex", survey_year = "2018", horizon= "1-Year", states = ["CA"]):
    if sensitive_attributes == "sex":
        # features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        # features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX']
        group = "RAC1P"
    else:
        features = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']
        group = "SEX"

    def adult_filter(data):
        """Mimic the filters in place for Adult data.
        Adult documentation notes: Extraction was done by Barry Becker from
        the 1994 Census database. A set of reasonably clean records was extracted
        using the following conditions:
        ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
        """
        df = data
        df = df[df['AGEP'] > 16]
        df = df[df['PINCP'] > 100]
        df = df[df['WKHP'] > 0]
        df = df[df['PWGTP'] >= 1]
        return df
    ACSIncome = folktables.BasicProblem(
        features=features,
        target='PINCP',
        target_transform=lambda x: x > 50000,
        group=group,
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
        )
    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey='person',
                                root_dir= data_root + "/NewAdult")
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)

    # print(f'features={features.shape}')
    # print(f'label={label.shape}')
    # print(f'group={group.shape}')

    X = pd.DataFrame(features, columns=ACSIncome.features)
    y = pd.Series(label)
    s = pd.Series(group, name=ACSIncome.group).to_frame()
    # print(f's={s}')
    if sensitive_attributes == "sex":
        s["SEX"] = s["SEX"] - 1
    elif sensitive_attributes == "race":
        s["RAC1P"] = (s["RAC1P"] == 1).astype(np.int)
    # X = pd.get_dummies(X, columns=ACSIncome.features)

    return X, y, s

def load_folktables_employment(sensitive_attributes="sex", survey_year = "2018", horizon= "1-Year", states = ["CA"]):
    if sensitive_attributes == "sex":
        # features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        features=['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','RAC1P',]
        group = "SEX"
    elif sensitive_attributes == "race":
        # features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        features=['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX',]
        group = "RAC1P"
    else:
        raise NotImplemented()
        
    def employment_filter(data):
        """
        Filters for the employment prediction task
        """
        df = data
        df = df[df['AGEP'] > 16]
        df = df[df['AGEP'] < 90]
        df = df[df['PWGTP'] >= 1]
        return df

    ACSEmployment = folktables.BasicProblem(
        features=features,
        target='ESR',
        target_transform=lambda x: x == 1,
        group=group,
        preprocess=lambda x: x,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey='person',
                                root_dir= data_root + "/NewAdult")
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)
    X = pd.DataFrame(features, columns=ACSEmployment.features)
    y = pd.Series(label)
    s = pd.Series(group, name=ACSEmployment.group).to_frame()
    if sensitive_attributes == "sex":
        s["SEX"] = s["SEX"] - 1
    elif sensitive_attributes == "race":
        s["RAC1P"] = (s["RAC1P"] == 1).astype(np.int)
    # X = pd.get_dummies(X, columns=ACSEmployment.features)
    return X, y, s

def load_ACS_I(sensitive_attributes="sex", survey_year = "2018", horizon= "1-Year", states = ["CA"], ratio=0.2):
    new_adult_x, new_adult_y, new_adult_s = load_folktables_income(sensitive_attributes, survey_year, horizon, states)
    train_data, test_data, train_target, test_target, \
    to_protect, to_protect_test = train_test_split(new_adult_x, new_adult_y, new_adult_s, test_size=ratio)

    return train_data.to_numpy(), train_target.to_numpy(), to_protect.to_numpy(), \
            test_data.to_numpy(), test_target.to_numpy(), to_protect_test.to_numpy()

def load_ACS_E(sensitive_attributes="sex", survey_year = "2018", horizon= "1-Year", states = ["CA"], ratio=0.2):
    new_adult_x, new_adult_y, new_adult_s = load_folktables_employment(sensitive_attributes, survey_year, horizon, states)
    train_data, test_data, train_target, test_target, \
    to_protect, to_protect_test = train_test_split(new_adult_x, new_adult_y, new_adult_s, test_size=ratio)

    return train_data.to_numpy(), train_target.to_numpy(), to_protect.to_numpy(), \
            test_data.to_numpy(), test_target.to_numpy(), to_protect_test.to_numpy()

def load_ACS(dataset, sensitive_attributes="sex", survey_year = "2018", horizon= "1-Year", states = ["CA"]):
    if dataset=="ACS_E":
        data_x, target, to_protect = load_folktables_employment(sensitive_attributes, survey_year, horizon, states)
    elif dataset=="ACS_I":
        data_x, target, to_protect = load_folktables_income(sensitive_attributes, survey_year, horizon, states)
    else:
        raise NotImplemented('Dataset {} does not exists'.format(dataset))
    
    return data_x.to_numpy(), target.to_numpy(), to_protect.to_numpy()

# re_adult = load_folktables_income(sensitive_attributes="sex")
# new_adult_x, new_adult_y, new_adult_s = re_adult
# print(new_adult_x.shape, new_adult_y.shape, new_adult_s.shape)

# re_employ = load_folktables_employment(sensitive_attributes="sex")
# new_adult_x, new_adult_y, new_adult_s = re_employ
# print(new_adult_x.shape, new_adult_y.shape, new_adult_s.shape)