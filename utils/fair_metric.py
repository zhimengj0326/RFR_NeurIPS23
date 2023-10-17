import torch
import torch.nn as nn
import torch.nn.functional as F

def dp(outputs, sens):
    ind_0 = (sens == 0).nonzero(as_tuple=False)
    ind_1 = (sens == 1).nonzero(as_tuple=False)

    dp_0 = torch.mean(outputs[ind_0])
    dp_1 = torch.mean(outputs[ind_1])
    dp_value = torch.abs(dp_0 - dp_1)

    return dp_value

def dp_smooth(outputs, sens):
    ind_0 = (sens == 0).nonzero(as_tuple=False)
    ind_1 = (sens == 1).nonzero(as_tuple=False)

    dp_0 = torch.mean(outputs[ind_0])
    dp_1 = torch.mean(outputs[ind_1])
    # dp_value = torch.abs(dp_0 - dp_1)

    return dp_0, dp_1


def eo(outputs, targets, sens):
    ind_label = (targets == 1).nonzero(as_tuple=False)
    outputs_label = outputs[ind_label]

    ind_0 = (sens[ind_label] == 0).nonzero(as_tuple=False)
    ind_1 = (sens[ind_label] == 1).nonzero(as_tuple=False)

    eo_0 = torch.mean(outputs_label[ind_0])
    eo_1 = torch.mean(outputs_label[ind_1])
    eo_value = torch.abs(eo_0 - eo_1)

    return eo_value