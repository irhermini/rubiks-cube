import torch.nn.functional as F




def oneHotEncoding(etats):
    return F.one_hot(etats.view(-1).long(), 6).view(-1, etats.shape[1]*6)