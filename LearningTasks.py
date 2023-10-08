import torch
import DataSetStates




def train(model, app, TrainLoader, ValidLoader, optimizer):
#def train(model, app, TrainLoader, lossFunction, optimizer):   
    model.train()
    LossesTrain = []
    LossesValid = []
    valuesHTrain = []
    valuesHValid = []    
    for i, (x, y) in enumerate(TrainLoader):
        
        x = x.to(app) #les états encodés
        y = y.to(app) #la cible avec quoi on va approximer h(s)
        hx = model(x)
        loss = torch.nn.MSELoss()(hx, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        LossesTrain.append(loss.item())
        valueHTrain = torch.mean(hx).item()
        valuesHTrain.append(valueHTrain)
        print("Entrainement : | Itération %d/%d | Perte : %.2f | Valeur Moyenne h %.3f"% (i + 1, len(TrainLoader), loss.item(), valueHTrain))
        
        
    with torch.no_grad():
        model.eval()
        for i, (x, y) in enumerate(ValidLoader):
    
            hx = model(x)
            loss = torch.nn.MSELoss()(hx, y)
            LossesValid.append(loss.item())
            valueHValid = torch.mean(hx).item()
            valuesHValid.append(valueHValid)
        
    meanLossTrain = sum(LossesTrain) / len(LossesTrain)
    meanValueHTrain = sum(valuesHTrain)/ len(valuesHTrain)
    meanLossValid = sum(LossesValid) / len(LossesValid)
    meanValueHValid = sum(valuesHValid) / len(valuesHValid)
    print("Validation : Perte Moyenne: %.2f | Valeur Moyenne h: %.3f"% (meanLossValid, meanValueHValid))
    return meanLossTrain, meanValueHTrain, meanLossValid, meanValueHValid
 #   r#eturn meanLossTrain, meanValueHTrain