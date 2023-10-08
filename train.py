import torch
from torch.optim.lr_scheduler import StepLR
from Model import *
import LearningTasks
from Cube import *
import time
import os
import DataSetStates



if __name__ == "__main__":
    app = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cube = Cube3()
    #path_network =  'save_models/numEtats 250 numEpochs 7000 lr 0.00100 batch 100 time 16092023-005352.pt'
    path_network = None
    saved_network = (path_network is not None)
    saved_data = False
    cubeNet = CubeNet(4)
    targetCubeNet = CubeNet(4)
    lr = 0.001
    lrDecay = 0.9995
    weightDecay = 0.00001
    numEpochs = 200
    numStates = 500
    Depth = 50
    batchSize = 400
    batchSizeValid = 1
    epoch_period = 50 #periode pour laquelle on va recharger les données
#    epoch_period = 10
    epoch_check = 20 #la période où on valorise le modèle via les loss moyens de la validation
    name = "numEtats %i numEpochs %i lr %.5f batch %i time %s"%(numStates, numEpochs, lr, batchSize, time.strftime('%d%m%Y-%H%M%S'))
    debut_entr = time.time()
    prop_train_valid = 0.8
    
    if path_network is not None:
        if os.path.isfile(path_network):
            cubeNet.load_state_dict(torch.load(path_network))
            targetCubeNet.load_state_dict(torch.load(path_network))
            print("Chargement du Réseau ...")
        else:
            raise ValueError("Réseau non trouvé !")
        
        
    else: 
        path_network = os.path.join("save_models", name) + ".pt"
    
    cubeNet.to(app)
    targetCubeNet.to(app)

    targetCubeNet.load_state_dict(cubeNet.state_dict())

    optimizer = torch.optim.Adam(cubeNet.parameters(), lr=lr, weight_decay=weightDecay)
    scheduler = StepLR(optimizer, step_size=1, gamma=lrDecay)
    optiLossValid = 0.1 #On enregistre le modèle lorsque la moyenne des loss de la validation sur epoch_check est inférieure à 0.1
    list_means = []
    
    
    if saved_data:
        try:
            DataSetStates.loadDataStates(numStates, [numEpochs - 1])
        except ValueError:
            raise(ValueError('Erreur de la fonction loadDataStates'))
            
            
    for epoch in range(1, numEpochs + 1): 
        if epoch % epoch_period == 1:
            debutPrep = time.time()
            idx = 0
            if not saved_data:
                oneHotStates, preparedData = DataSetStates.prepareDataStates(cube, numStates * epoch_period, Depth)
            else:
                JDD = DataSetStates.loadDataStates(numStates, range(epoch - 1, epoch + epoch_period - 1) )
            dureePrep = time.time() - debutPrep
            print("Durée de préparation des données : %.3f secondes" % (dureePrep))
            
        debutEpoch = time.time()
        if not saved_data: 
            preparedsubData = [data[idx*numStates:(idx + 1) * numStates] for data in preparedData]        
            oneHotsubStates = oneHotStates[idx*numStates:(idx + 1) * numStates]
        else:
            preparedsubData = JDD[idx][1]
            oneHotsubStates = JDD[idx][0]
        cible = DataSetStates.createTrainingData(cube, preparedsubData, targetCubeNet, app)
        
        oneHotsubStatesTrain = oneHotsubStates[:int(prop_train_valid * numStates)]
        oneHotsubStatesValid = oneHotsubStates[int(prop_train_valid * numStates):]
        cibleTrain = cible[:int(prop_train_valid * numStates)]
        cibleValid = cible[int(prop_train_valid * numStates):]
        
        idx += 1
        DataSetTrain = DataSetStates.CubeDataSet(oneHotsubStatesTrain, cibleTrain)
        DataSetValid = DataSetStates.CubeDataSet(oneHotsubStatesValid, cibleValid)
        #DataSetTrain = DataSetStates.CubeDataSet(oneHotsubStates, cible)
        trainLoader = torch.utils.data.DataLoader(DataSetTrain, batch_size = batchSize, shuffle = True, num_workers = 0)
        validLoader = torch.utils.data.DataLoader(DataSetValid, batch_size = batchSizeValid, shuffle = False)
        meanLossTrain, meanValueHTrain, meanLossValid, meanValueHValid = LearningTasks.train(cubeNet, app, trainLoader, validLoader, optimizer)
        #meanLossTrain, meanValueHTrain = LearningTasks.train(cubeNet, app, trainLoader, torch.nn.MSELoss(), optimizer)
        scheduler.step()
        
        dureeEpoch = time.time() - debutEpoch
        print("Epoch: %d/%d | Durée de l'Epoch : %.3f secondes" %
              (epoch, numEpochs, dureeEpoch))
        list_means.append(meanLossValid)
        
        if epoch == 100 and not saved_network:
            targetCubeNet.load_state_dict(cubeNet.state_dict())
            
        if epoch % epoch_check == 0:
            if sum(list_means)/epoch_check < optiLossValid: 
                targetCubeNet.load_state_dict(cubeNet.state_dict())
                print("Enregistremement du modèle")
                torch.save(cubeNet.state_dict(), path_network)
            else:
                print("La perte est très grande, pas d'enregistrement")
            list_means=[]
        
        
    
    duree_entr = time.time() - debut_entr
    print("Le temps d'entrainement est de %i heures, %i minutes et %i secondes"
        % (duree_entr // 3600, duree_entr // 60 % 60, duree_entr % 60)
    )