import torch
from torch.utils.data import Dataset
from oneHotEncoding import oneHotEncoding
import pandas as pd
import os
from utils import saveFile, loadFile
import time

def prepareDataStates(Cube, numEtats, Depth):
    
    etats = Cube.generateStates(0, Depth, numEtats)
    Buts = Cube.areSolved(etats)
    EtatsSuivants, ButsEtatsSuivants = Cube.explorerEtatsSuivants(etats)
    encodedStates = oneHotEncoding(etats).detach()
    
    return encodedStates , [etats, Buts, EtatsSuivants, ButsEtatsSuivants]
    
    
def saveDataStates(Cube, numEtats, numFiles, Depth):
#     numEtats = 0
     name_dir = os.path.join("save_data", str(numEtats))
     if not os.path.exists(name_dir):
         os.mkdir(name_dir)
     
     for i in range(numFiles):
         encodedStates , [etats, Buts, EtatsSuivants, ButsEtatsSuivants] = prepareDataStates(Cube, numEtats, Depth)
         
         now = time.strftime('%d%m%Y-%H%M%S')
         
         dic = {'Etats encodés' : encodedStates, 'Etats' : etats, 'Buts' : Buts, 'Etats Suivants' : EtatsSuivants, 'Buts Etats Suivants' : ButsEtatsSuivants}
         saveFile(os.path.join(name_dir, "numEtats%iDepth%iDate%s_%i.p"%(numEtats, Depth, now, i)), dic)
         print("Fichier %i sur %i enregistré"%(i + 1, numFiles))
         
def loadDataStates(numEtats, L):
    
    name_dir = os.path.join("save_data", str(numEtats))
    if not os.path.exists(name_dir):
        raise ValueError("Pas de données trouvés pour le nombre d'états donné")
    elif len(os.listdir(name_dir)) <= max(L):
        raise ValueError("Le nombre de fichiers demandé n'est pas disponible en base de données")
    else: 
        data = []
        for i in L:
            file = os.listdir(name_dir)[i]
            dic = loadFile(name_dir + '/' + file)
            encodedStates, etats, Buts, EtatsSuivants, ButsEtatsSuivants = dic.values()
            data.append((encodedStates, [etats, Buts, EtatsSuivants, ButsEtatsSuivants]))
        return data

def createTrainingData(Cube, preparedData, model, app):

    etats, Buts, EtatsSuivants, ButsEtatsSuivants = preparedData
    EtatsSuivantsLoins = EtatsSuivants[~ButsEtatsSuivants]
    
    EtatsSuivantsLoinsoneHot = oneHotEncoding(EtatsSuivantsLoins).to(app)
    
    HEtatsSuivantsLoins = model(EtatsSuivantsLoinsoneHot)
    
    cible = torch.zeros(EtatsSuivants.shape[:2])
    cible[~ButsEtatsSuivants] = HEtatsSuivantsLoins #la cible des états suivants est égale à h(s') si les états suivants ne sont pas résolus, et 0 si résolus
    cible[ButsEtatsSuivants] = 0 #h(s') est nulle si les états suivants sont résolus
    
    cible = cible.min(1).values + 1 #h(s) = 1 + min h(s') ou s' les états suivants de l'état s
    
    cible[Buts] = 0 #h(s) = 0 si l'état est résolu
    
    return cible.detach()


             
class CubeDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, i):
        return self.data[i], self.labels[i]
        
    def __len__(self):
        return len(self.data)
