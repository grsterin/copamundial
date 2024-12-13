import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, n_channel, n_head, n_dim, n_out, activation = "identity"):
        super(MHA, self).__init__()
        activations = {"relu" : nn.ReLU(), "sigmoid" : nn.Sigmoid(), "tanh": nn.Tanh()}
        self.Wq = nn.Parameter(torch.randn(n_head, n_dim, n_channel, dtype = torch.float32))
        self.Wk = nn.Parameter(torch.randn(n_head, n_dim, n_channel, dtype = torch.float32))
        self.Wv = nn.Parameter(torch.randn(n_head, n_dim, n_channel, dtype = torch.float32))
        self.drop = nn.Dropout(p=0.2)
        self.activation = activations[activation] if activation in activations else nn.Identity()
        self.n_channel  = nn.Parameter(torch.tensor(n_channel, dtype = torch.float32), requires_grad = False)
        self.Wo = nn.Parameter(torch.randn(n_head * n_channel, n_out))
        
    def forward(self, X):
        # X is batch x n_seq = 100 x n_dim = 1
        X1 = X.unsqueeze(1)
        q  = torch.matmul(X1, self.Wq) # batch x n_head x n_seq x n_channel
        k  = torch.matmul(X1, self.Wk)
        v  = torch.matmul(X1, self.Wv)
        
        att = F.softmax(torch.matmul(q, torch.transpose(k, 2, 3)) / torch.sqrt(self.n_channel), dim = -1) 
        v = torch.matmul(att, v) # batch x n_head x n_seq x n_channel
        v = self.drop(v)
        vc = torch.concatenate(torch.unbind(v, 1), axis = -1)
        return self.activation(torch.matmul(vc, self.Wo))
        
        
class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.proj1 = nn.Linear(1, 4) 
        self.mha1  = nn.MultiheadAttention(4, 2, batch_first = True)
        self.drop1 = nn.Dropout(p=0.2)
        self.act   = torch.tanh
        self.mha2  = nn.MultiheadAttention(4, 2, batch_first = True)
        self.proj2 = nn.Linear(4, 1)
        
    def forward(self, x):
        """
        input : batch x nseq x 1
        => batch x nseq x 10
        => batch x nseq x 10
        => batch x nseq x 10
        => batch x nseq x 1
        """
        out = self.drop1(self.act(self.proj1(x)))
        out = out + self.mha1(out, out, out)[0]
        out = self.act(out)
        out = out + self.mha2(out, out, out)[0]
        out = self.act(out)
        return self.proj2(out)
    
            
class AttentionModel2(nn.Module):
    def __init__(self, dims = 20, activation = torch.sigmoid, nhead = 4,
                num_layers = 3):
        super(AttentionModel2, self).__init__()
        self.proj1 = nn.Linear(1, dims) 
        self.proj2 = nn.Linear(dims, 1)
        
        self.drop1 = nn.Dropout(p=0.2)
        self.act   = activation
        
        """
        self.mha1  = nn.MultiheadAttention(50, 5, batch_first = True)
        self.mha2  = nn.MultiheadAttention(50, 5, batch_first = True)
        self.mha3  = nn.MultiheadAttention(50, 5, batch_first = True)
        """
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dims, nhead=nhead, activation = torch.tanh)
        self.tencoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        """
        input : batch x nseq x 1
        => batch x nseq x 10
        => batch x nseq x 10
        => batch x nseq x 10
        => batch x nseq x 1
        """
        out = self.drop1(self.act(self.proj1(x)))
        """
        out = self.mha1(out1, out1, out1)[0]
        out = self.act(out)
        out = self.mha2(out, out, out)[0]
        out = self.act(out)
        """
        out = self.tencoder(out)
        return self.proj2(self.act(out))
    
            
class AttentionModel3(nn.Module):
    def __init__(self, dims = [256, 128, 64], activation = nn.Tanh, nhead = 4,
                num_layers = 3, svd_dim=200, **kwargs):
        super(AttentionModel3, self).__init__()
        
        mods = []
        prev_dim = svd_dim
        for d in dims:
            mods.append(nn.Linear(prev_dim, d))
            mods.append(nn.Dropout(p=0.2))
            mods.append(activation())
            prev_dim = d
        mods.append(nn.Linear(prev_dim, svd_dim))
        self.mod = nn.Sequential(*mods)
        return
        
    def forward(self, x):
        x = x.squeeze(-1)
        x = self.mod(x)
        return x.unsqueeze(-1)
        