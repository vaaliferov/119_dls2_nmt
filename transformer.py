import torch
import numpy as np

class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.device = device
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.fc_q = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_k = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_v = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_o = torch.nn.Linear(hid_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        s = lambda x: x.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        Q, K, V = s(self.fc_q(query)), s(self.fc_k(key)), s(self.fc_v(value))
        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        if mask is not None: energy = energy.masked_fill(mask == 0, -1e10)
        attn = torch.softmax(energy, dim=-1); x = torch.matmul(self.dropout(attn), V)
        x = x.permute(0,2,1,3).contiguous().view(batch_size, -1, self.hid_dim)
        return self.fc_o(x), attn

class PositionwiseFeedforward(torch.nn.Module):
    
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hid_dim, pf_dim)
        self.fc2 = torch.nn.Linear(pf_dim, hid_dim)
    
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class EncoderLayer(torch.nn.Module):
    
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.ff_norm = torch.nn.LayerNorm(hid_dim)
        self.self_attn_norm = torch.nn.LayerNorm(hid_dim)
        self.ff = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.self_attn = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        
    def forward(self, x, mask):
        ax, self_attn = self.self_attn(x, x, x, mask)
        x = self.self_attn_norm(x + self.dropout(ax))
        x = self.ff_norm(x + self.dropout(self.ff(x)))
        return x, self_attn

class DecoderLayer(torch.nn.Module):
    
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.ff_norm = torch.nn.LayerNorm(hid_dim)
        self.enc_attn_norm = torch.nn.LayerNorm(hid_dim)
        self.self_attn_norm = torch.nn.LayerNorm(hid_dim)
        self.ff = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.enc_attn = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.self_attn = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        
    def forward(self, x, mask, src_enc, src_mask):
        ax, self_attn = self.self_attn(x, x, x, mask)
        x = self.self_attn_norm(x + self.dropout(ax))
        ax, enc_attn = self.enc_attn(x, src_enc, src_enc, src_mask)
        x = self.enc_attn_norm(x + self.dropout(ax))
        x = self.ff_norm(x + self.dropout(self.ff(x)))
        return x, self_attn, enc_attn

class Encoder(torch.nn.Module):
    
    def __init__(self, inp_dim, hid_dim, pf_dim, 
                 n_layers, n_heads, dropout, max_len, device):
        
        super().__init__()
        self.device = device
        self.dropout = torch.nn.Dropout(dropout)
        self.pos_embs = torch.nn.Embedding(max_len, hid_dim)
        self.tok_embs = torch.nn.Embedding(inp_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        layer = lambda: EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
        self.layers = torch.nn.ModuleList([layer() for _ in range(n_layers)])
        
    def forward(self, x, mask):
        self_attn_list = []
        batch_size, x_len = x.shape
        pos = torch.arange(x_len).unsqueeze(0)
        pos = pos.repeat(batch_size, 1).to(self.device)
        x = self.tok_embs(x) * self.scale
        x = self.dropout(x + self.pos_embs(pos))
        for layer in self.layers: 
            x, self_attn = layer(x, mask)
            self_attn_list.append(self_attn)
        return x, self_attn_list

class Decoder(torch.nn.Module):
    
    def __init__(self, out_dim, hid_dim, pf_dim, 
                 n_layers, n_heads, dropout, max_len, device):
        
        super().__init__()
        self.device = device
        self.dropout = torch.nn.Dropout(dropout)
        self.fc_out = torch.nn.Linear(hid_dim, out_dim)
        self.pos_embs = torch.nn.Embedding(max_len, hid_dim)
        self.tok_embs = torch.nn.Embedding(out_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        layer = lambda: DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
        self.layers = torch.nn.ModuleList([layer() for _ in range(n_layers)])
        
    def forward(self, x, mask, src_enc, src_mask):
        batch_size, x_len = x.shape
        self_attn_list, enc_attn_list = [], []
        pos = torch.arange(x_len).unsqueeze(0)
        pos = pos.repeat(batch_size, 1).to(self.device)
        x = self.tok_embs(x) * self.scale
        x = self.dropout(x + self.pos_embs(pos))
        for layer in self.layers: 
            x, self_attn, enc_attn = layer(x, mask, src_enc, src_mask)
            self_attn_list.append(self_attn); enc_attn_list.append(enc_attn)
        return self.fc_out(x), self_attn_list, enc_attn_list

class Model(torch.nn.Module):
    
    def __init__(self, n_layers, n_heads, max_len, 
                 inp_dim, out_dim, hid_dim, pf_dim, 
                 dropout, bos_idx, eos_idx, pad_idx, device):
        
        super().__init__()
        self.device = device
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.max_len = max_len
        
        self.encoder = Encoder(
            inp_dim, hid_dim, pf_dim, 
            n_layers, n_heads, dropout, 
            max_len, device).to(device)
        
        self.decoder = Decoder(
            out_dim, hid_dim, pf_dim, 
            n_layers, n_heads, dropout, 
            max_len, device).to(device)
        
        for p in self.parameters():
            if p.dim() > 1: torch.nn.init.xavier_uniform_(p)
    
    @staticmethod
    def make_mask(x, pad_idx):
        return (x != pad_idx).unsqueeze(1).unsqueeze(2)
    
    @staticmethod
    def make_tril_mask(x, pad_idx):
        pad_mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        sub_mask = torch.ones((x.shape[1], x.shape[1]))
        sub_mask = torch.tril(sub_mask).bool()
        return pad_mask & sub_mask.to(x.device)
    
    def forward(self, src, trg):
        src_mask = self.make_mask(src, self.pad_idx)
        trg_mask = self.make_tril_mask(trg, self.pad_idx)
        src_enc, enc_self_attn = self.encoder(src, src_mask)
        out, dec_self_attn, enc_dec_attn = self.decoder(trg, trg_mask, src_enc, src_mask)
        return out, enc_self_attn, dec_self_attn, enc_dec_attn
    
    def greedy_generate(self, src):
        self.eval()
        torch.set_grad_enabled(False)
        src = torch.tensor(src, dtype=torch.long)
        src = src.unsqueeze(0).to(self.device)
        src_mask = self.make_mask(src, self.pad_idx)
        src_enc, enc_self_attn = self.encoder(src, src_mask)
        trg = torch.tensor([[self.bos_idx]]).to(self.device)
        for i in range(self.max_len):
            trg_mask = self.make_tril_mask(trg, self.pad_idx)
            out, dec_self_attn, dec_enc_attn = self.decoder(trg, trg_mask, src_enc, src_mask)
            pred_token = out.argmax(-1)[:,-1].item()
            trg = torch.tensor([trg.tolist()[0] + [pred_token]]).to(self.device)
            if pred_token == self.eos_idx: break
        return trg.tolist()[0], enc_self_attn, dec_self_attn, dec_enc_attn
    
    def beam_generate(self, src, k=4):
        self.eval()
        torch.set_grad_enabled(False)
        src = torch.tensor(src, dtype=torch.long)
        src = src.unsqueeze(0).to(self.device)
        src_mask = self.make_mask(src, self.pad_idx)
        src_enc, enc_self_attn = self.encoder(src, src_mask)
        src_enc = src_enc.repeat(k, 1, 1)
        trg = torch.tensor([[self.bos_idx]] * k)
        probs = torch.tensor([[1.]] * k)
        eos = torch.tensor([False] * k)
        for i in range(self.max_len):
            trg_mask = self.make_tril_mask(trg, self.pad_idx)
            out, dec_self_attn, dec_enc_attn = self.decoder(trg, trg_mask, src_enc, src_mask)
            p, t = torch.topk(torch.softmax(out, dim=-1), k=k, dim=-1)
            p, t = p[:,-1,:], t[:,-1,:]
            m = (t == self.eos_idx) | (eos.reshape(-1,1).repeat(1,k))
            p[m], t[m] = 1., self.eos_idx
            probs = torch.hstack((probs.repeat(k,1), p.T.reshape(-1,1)))
            trg = torch.hstack((trg.repeat(k,1), t.T.reshape(-1,1)))
            unique, idx = np.unique(trg.numpy(), axis=0, return_index=True)
            probs, trg = probs[idx], trg[idx]
            idx = torch.argsort(-torch.prod(probs, dim=-1))[:k]
            probs, trg = probs[idx], trg[idx]
            eos |= (trg[:,-1] == self.eos_idx)
            if torch.all(eos): break;
        return trg.tolist(), enc_self_attn, dec_self_attn, dec_enc_attn
