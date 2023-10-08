import torch
import numpy as np




class Cube3:
    def __init__(self):
        self.ListeActions = ["U","U'","F","F'","L","L'","D","D'","B","B'","R","R'"]
        self.ListeFaces  = ["Haut", "Avant", "Gauche", "Bas", "Arrière", "Droit"]
        self.Couleurs = ['white', 'red', 'green', 'yellow', 'orange', 'blue']
        self.Sens = ["Horaire", "Trigonomètrique"]
        self.Actions = {self.ListeActions[i] : (i, "Tourner la face " + self.ListeFaces[i//2] + " au sens " + self.Sens[i%2]) for i in range(0, 12)}
        self.EtatResolu = self.getSolvedState()
        self.MatEtatsSuivants = self.getMatNextStates()
        
    def getSolvedState(self):
        etat = torch.zeros(54, dtype=torch.uint8)
        for i in range(6):
            etat[9 * i: 9 * (i + 1)] = i

        return etat
    
    def getMatNextStates(self):
        MatCubeResolu = np.arange(54)
        MatEtatsSuivants = np.tile(MatCubeResolu, (12, 1))
        IndicesVoisins = np.zeros([6,12], dtype = np.uint8)
        IndicesVoisins[0] = np.array([11, 10,  9, 20, 19, 18, 38, 37, 36, 47, 46, 45]) #IndicesVoisins[i] : ce sont les indices qui voisinent celle de la Face i comptées au sens horaire
        IndicesVoisins[1] = np.array([26, 23, 20,  6,  7,  8, 45, 48, 51, 29, 28, 27]) #et qui seront impactés par une rotation de cette Face
        IndicesVoisins[2] = np.array([27, 30, 33, 44, 41, 38,  0,  3,  6,  9, 12, 15])
        IndicesVoisins[3] = np.array([42, 43, 44, 24, 25, 26, 15, 16, 17, 51, 52, 53])
        IndicesVoisins[4] = np.array([53, 50, 47,  2,  1,  0, 18, 21, 24, 33, 34, 35])
        IndicesVoisins[5] = np.array([ 8,  5,  2, 36, 39, 42, 35, 32, 29, 17, 14, 11])

        for Action in range(0,12):
            CubeApresRotation = MatCubeResolu.copy()
            Face = Action // 2
            FaceToRotate = (MatCubeResolu[Face* 9:(Face + 1) * 9]).reshape(3,3)
            IndicesVoisinsApresRotation = IndicesVoisins[Face]
            if Action %2 ==0 :  #rotation au sens horaire
                FaceToRotate = np.rot90(FaceToRotate, 3)
                IndicesVoisinsApresRotation = np.roll(IndicesVoisinsApresRotation,3)
            else: #rotation au sens trigonométrique
                FaceToRotate = np.rot90(FaceToRotate, 1)
                IndicesVoisinsApresRotation = np.roll(IndicesVoisinsApresRotation, -3)
            CubeApresRotation[9 * Face : 9* (Face + 1)] = FaceToRotate.flatten()
            CubeApresRotation[IndicesVoisins[Face]] = IndicesVoisinsApresRotation
            
            MatEtatsSuivants[Action] = CubeApresRotation
        return torch.tensor(MatEtatsSuivants, dtype=torch.int64)
    
    def doAction(self, action, etat=None):
        assert action in self.Actions

        if etat is None:
            etat = self.EtatResolu

        etat_suivant = etat[self.MatEtatsSuivants[self.Actions[action][0]]]

        return etat_suivant
    
    def generateState(self, numActions):
        Actions = np.random.randint(0,12, numActions)
        etat = self.EtatResolu.clone()
        for action in Actions:
            etat = self.doAction(self.ListeActions[action], etat)
        
        return etat
        
        
    def generateStates(self, minDepth, maxDepth, nbreEtats):
        Depths = np.random.randint(minDepth, maxDepth + 1, nbreEtats)
        Etats = torch.zeros([nbreEtats, 54], dtype = torch.uint8)
        EtatResolu = self.EtatResolu
        for n in range(nbreEtats):
            Etat = self.generateState(Depths[n])
            Etats[n] = Etat
        return Etats

        
    def areSolved(self, etats):
        return torch.all(etats == self.EtatResolu, 1)

    def isSolved(self, etat):
        return torch.equal(etat, self.EtatResolu)
        
    def explorerEtatsSuivants(self, etats):
        nbreEtats = etats.shape[0]
        MatActionEtats = torch.as_tensor(np.tile(np.arange(0, 12,dtype=np.int64), nbreEtats))
        IdxActionEtatsSuivants = self.MatEtatsSuivants.index_select(0, MatActionEtats)
        EtatsSuivants = etats.repeat_interleave(12, dim = 0).gather(1, IdxActionEtatsSuivants)
        EtatsSuivants = EtatsSuivants.view(etats.shape[0], 12, -1)

        Buts = self.areSolved(EtatsSuivants.view(-1, 54)).view(-1, 12)

        return EtatsSuivants, Buts    
    
    
