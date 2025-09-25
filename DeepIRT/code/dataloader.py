import torch
import torch.utils.data as Data
from readdata import DataReader
import torch.utils.data.dataset as ds

def getDataLoader(batch_size, num_of_questions, max_step):
    handle = DataReader('../data/assist2015/assist2015_train.txt',
                        '../data/assist2015/assist2015_test.txt', max_step,
                        num_of_questions)
    train, vali = handle.getTrainData()
    dtrain = torch.tensor(train.astype(int).tolist(), dtype=torch.long)
    dvali = torch.tensor(vali.astype(int).tolist(), dtype=torch.long)
    dtest = torch.tensor(handle.getTestData().astype(int).tolist(),
                         dtype=torch.long)
    trainLoader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    valiLoader = Data.DataLoader(dvali, batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return trainLoader, valiLoader, testLoader