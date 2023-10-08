import time
import torch
from oneHotEncoding import oneHotEncoding


class Noeud:
    def __init__(self, etat, prof, cout, parent, mvtParent, Resolu):
         self.etat = etat
         self.prof = prof
         self.cout = cout
         self.parent = parent
         self.mvtParent = mvtParent 
         self.Resolu = Resolu 
         
    def __hash__(self):
        return hash(self.etat.numpy().data.tobytes())
        

def AEtoileRech(etatinitial, Cube, poids, fonction, app, maxItr, numMaxNoeuds):
    fonction.to(app)
    NoeudsFermes = dict()
    racine = Noeud(etatinitial, 0, 0, None, None, Cube.isSolved(etatinitial))
    NoeudsOuverts = [(racine.cout, id(racine), racine)]
    Itr = 1
    numNoeuds = 1
    Resolu = False
    
    InstDebut = time.time()
    with torch.no_grad():
        while Itr <= maxItr and not Resolu: 
            InstDebutItr = time.time()
            NoeudsOuverts.sort()
            numGet = min(len(NoeudsOuverts) , numMaxNoeuds)
            NoeudsActuels = []
            for i in range(numGet):
                noeud = NoeudsOuverts.pop(0)[2]
                NoeudsActuels.append(noeud)
                if noeud.Resolu:
                    Resolu = True
                    NoeudResolu = noeud
                    
                    
            if not Resolu:
                etatsActuels = [noeud.etat for noeud in NoeudsActuels]
                etatsActuels = torch.stack(etatsActuels)
                
                etatsEnfants, butsEtatsEnfants = Cube.explorerEtatsSuivants(etatsActuels)
                
                enfants = []
                profs = []
                
                for i in range(len(NoeudsActuels)):
                     parent = NoeudsActuels[i]
                     for j in range(0,12):
                         profs.append(parent.prof + 1)
                         enfant = Noeud(etatsEnfants[i][j], parent.prof + 1, 0, parent, Cube.ListeActions[j], butsEtatsEnfants[i][j])
                         enfants.append(enfant)
                         
                IdxAjoutsNoeuds = []
                
                for i, enfant in enumerate(enfants):
                    if hash(enfant) in NoeudsFermes:
                        if NoeudsFermes[hash(enfant)].prof > enfant.prof:
                            enfanttrouve = NoeudsFermes.pop(hash(enfant))
                            enfanttrouve.prof = enfant.prof
                            enfanttrouve.parent = enfant.parent
                            enfanttrouve.mvtParent = enfant.mvtParent
                            enfants[i] = enfanttrouve
                            IdxAjoutsNoeuds.append(i)
                            NoeudsFermes[hash(enfant)] = enfanttrouve
                    else:
                        IdxAjoutsNoeuds.append(i)
                        NoeudsFermes[hash(enfant)] = enfant
                        
                enfants = [enfants[i] for i in IdxAjoutsNoeuds]
                profs = torch.tensor([profs[i] for i in IdxAjoutsNoeuds])
                
                
                etatsEnfants = etatsEnfants.reshape(etatsEnfants.shape[0]*etatsEnfants.shape[1], etatsEnfants.shape[2])
                etatsEnfants = etatsEnfants[IdxAjoutsNoeuds]
                etatsEnfants = oneHotEncoding(etatsEnfants).to(app)
                
                hValeurs = fonction(etatsEnfants).cpu()
                
                if len(hValeurs) > 0: 
                    minH = min(hValeurs)
                    
                couts = poids * profs + hValeurs
                
                if len(couts) > 0:
                    minC = min(couts)
                    
                for i in range(len(couts)):
                    enfants[i].cout = float(couts[i])
                    NoeudsOuverts.append((enfants[i].cout, id(enfants[i]), enfants[i]))
                numNoeuds += len(enfants)
                
                Itr += 1
                print("Itération numéro : %i, Meilleure Valeur de H : %.3f, Coût minimal des enfants : %.3f Durée d'itération : %3f secondes"
                %(Itr, minH, minC, time.time() - InstDebutItr))
    
    DureeRecherche = time.time() - InstDebut
    if Resolu: 
        
        mvts = []
        noeud = NoeudResolu
        
        while noeud.prof > 0: 
            mvts = [noeud.mvtParent] + mvts
            noeud = noeud.parent
            
            
            
    else:
        mvts = None
                         
    return mvts , numNoeuds, Itr, Resolu, DureeRecherche         
    