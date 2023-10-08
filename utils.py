import pickle





def saveFile(path, obj):
    with open(path,'wb') as file :
        pickle.Pickler(file).dump(obj)
    
def loadFile(path):
    with open(path,'rb') as file :
        obj = pickle.Unpickler(file).load()
    return obj
