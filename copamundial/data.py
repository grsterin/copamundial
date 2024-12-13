import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch 

class Data(Dataset):
    def __init__(self, matchfile, no_matches, Xa, Xb, nA, nB, rev = False):
        self.no_matches = no_matches
        self.matchdf = pd.read_csv(matchfile, sep = "\t")
        self.rev = rev
        if "score" in self.matchdf.columns:
            self.matchdf = self.matchdf.sort_values(by = "score", ascending = False).reset_index(drop = True)[: no_matches]
        else:
            self.matchdf = self.matchdf.loc[: no_matches, :]
            # for compatibility
            self.matchdf["score"] = 1
        print(self.matchdf)
        self.nA = nA
        self.nB = nB
        self.Xa = Xa
        self.Xb = Xb
        
    def __len__(self):
        return self.no_matches
    
    def __getitem__(self, idx):
        pa, pb, _= self.matchdf.iloc[idx, :].values
        if self.rev:
            temp = pa
            pa   = pb
            pb   = temp
            
        ia, ib = self.nA[pa], self.nB[pb]
        return torch.tensor(self.Xa[ia], dtype = torch.float32).unsqueeze(-1), torch.tensor(self.Xb[ib], dtype = torch.float32).unsqueeze(-1)

    

class CuratedData(Dataset):
    def __init__(self, Xa, Xb):
        self.Xa = Xa
        self.Xb = Xb
        
    def __len__(self):
        return self.Xa.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.Xa[idx], dtype = torch.float32).unsqueeze(-1), torch.tensor(self.Xb[idx], dtype = torch.float32).unsqueeze(-1)
        
        
class PredictData(Dataset):
    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype = torch.float32).unsqueeze(-1)