import os
import torch
import random
import numpy as np
from sklearn.metrics import log_loss

def seed_everything(seed=47):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def evaluate(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    score = 0
    for i in range(y_true.shape[1]):
        score_ = log_loss(y_true[:, i], y_pred[:, i])
        score += score_ 
    score /= y_true.shape[1]
    return score
