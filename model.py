import torch
import numpy as np
from config import *

class ConcreteGate(torch.nn.Module):
    def __init__(self, shape, device):
        super().__init__()
        self.eps = GATE_EPS
        self.device = device
        self.hard = GATE_HARD
        self.l0_penalty = GATE_PENALTY
        self.local_rep = GATE_LOCAL_REP
        self.temperature = GATE_TEMPERATURE
        self.stretch_limits = GATE_STRETCH_LIMITS
        log_a = torch.zeros(shape).to(device)
        self.log_a = torch.nn.Parameter(log_a)
    
    def get_penalty(self, values, sum_dim):
        low, high = self.stretch_limits
        p_open = torch.sigmoid(self.log_a - self.temperature * torch.log(torch.tensor(-low / high)))
        p_open = torch.clamp(p_open, self.eps, 1.0 - self.eps)
        if self.local_rep:
            p_open = torch.add(p_open, torch.zeros_like(values).to(self.device))
            return self.l0_penalty * torch.mean(torch.sum(p_open, sum_dim))
        return self.l0_penalty * torch.sum(p_open)
        
    def get_gates(self, values, is_train):
        low, high = self.stretch_limits
        clear_concrete = torch.sigmoid(self.log_a)
        shape = self.log_a.shape if values == None else values.shape
        noise = torch.zeros(shape).uniform_(self.eps, 1.0 - self.eps).to(self.device)
        noisy_concrete = torch.sigmoid((torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature)
        concrete = noisy_concrete if is_train else clear_concrete
        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete, 0, 1)
        if self.hard:
            hard_concrete = (torch.greater(clipped_concrete, 0.5)).float()
            clipped_concrete = clipped_concrete + (hard_concrete - clipped_concrete).detach()
        return clipped_concrete
    
    def forward(self, values, is_train, sum_dim):
        self.penalty = self.get_penalty(values, sum_dim)
        self.gates = self.get_gates(values, is_train)
        return values * self.gates.to(self.device)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, pruning, device):
        super().__init__()
        self.device = device
        self.pruning = pruning
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.fc_q = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_k = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_v = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_o = torch.nn.Linear(hid_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        if self.pruning: self.gate = ConcreteGate((1, n_heads, 1, 1), device)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        s = lambda x: x.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        Q, K, V = s(self.fc_q(query)), s(self.fc_k(key)), s(self.fc_v(value))
        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        if mask is not None: energy = energy.masked_fill(mask == 0, -1e10)
        attn = torch.softmax(energy, dim=-1); x = torch.matmul(self.dropout(attn), V)
        if self.pruning: x = self.gate(x, self.training, sum_dim=(1,3))
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
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, pruning, device):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.ff_norm = torch.nn.LayerNorm(hid_dim)
        self.self_attn_norm = torch.nn.LayerNorm(hid_dim)
        self.ff = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.self_attn = MultiHeadAttention(hid_dim, n_heads, dropout, pruning, device)
        
    def forward(self, x, mask):
        ax, self_attn = self.self_attn(x, x, x, mask)
        x = self.self_attn_norm(x + self.dropout(ax))
        x = self.ff_norm(x + self.dropout(self.ff(x)))
        return x, self_attn

class DecoderLayer(torch.nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, pruning, device):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.ff_norm = torch.nn.LayerNorm(hid_dim)
        self.enc_attn_norm = torch.nn.LayerNorm(hid_dim)
        self.self_attn_norm = torch.nn.LayerNorm(hid_dim)
        self.ff = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.enc_attn = MultiHeadAttention(hid_dim, n_heads, dropout, pruning, device)
        self.self_attn = MultiHeadAttention(hid_dim, n_heads, dropout, pruning, device)
        
    def forward(self, x, mask, src_enc, src_mask):
        ax, self_attn = self.self_attn(x, x, x, mask)
        x = self.self_attn_norm(x + self.dropout(ax))
        ax, enc_attn = self.enc_attn(x, src_enc, src_enc, src_mask)
        x = self.enc_attn_norm(x + self.dropout(ax))
        x = self.ff_norm(x + self.dropout(self.ff(x)))
        return x, self_attn, enc_attn

class Encoder(torch.nn.Module):
    
    def __init__(self, inp_dim, hid_dim, pf_dim, 
                 n_layers, n_heads, dropout, max_len, pruning, device):
        
        super().__init__()
        self.device = device
        self.dropout = torch.nn.Dropout(dropout)
        self.pos_embs = torch.nn.Embedding(max_len, hid_dim)
        self.tok_embs = torch.nn.Embedding(inp_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        layer = lambda: EncoderLayer(hid_dim, n_heads, pf_dim, dropout, pruning, device)
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
                 n_layers, n_heads, dropout, max_len, pruning, device):
        
        super().__init__()
        self.device = device
        self.dropout = torch.nn.Dropout(dropout)
        self.fc_out = torch.nn.Linear(hid_dim, out_dim)
        self.pos_embs = torch.nn.Embedding(max_len, hid_dim)
        self.tok_embs = torch.nn.Embedding(out_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        layer = lambda: DecoderLayer(hid_dim, n_heads, pf_dim, dropout, pruning, device)
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
    
    def __init__(self, inp_dim, out_dim, device):
        
        super().__init__()
        self.device = device
        self.bos_idx = BOS_IDX
        self.eos_idx = EOS_IDX
        self.pad_idx = PAD_IDX
        self.max_len = MAX_TOKS_NUM
        
        self.encoder = Encoder(
            inp_dim, HID_DIM, PF_DIM, N_LAYERS, N_HEADS, 
            DROPOUT, MAX_TOKS_NUM, PRUNING, device).to(device)
        
        self.decoder = Decoder(
            out_dim, HID_DIM, PF_DIM, N_LAYERS, N_HEADS, 
            DROPOUT, MAX_TOKS_NUM, PRUNING, device).to(device)
        
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
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_penalty(self):
        gates = [layer.self_attn.gate for layer in self.encoder.layers]
        gates += [layer.self_attn.gate for layer in self.decoder.layers]
        gates += [layer.enc_attn.gate for layer in self.decoder.layers]
        return torch.tensor([gate.penalty for gate in gates]).mean()
    
    def get_gates(self):
        f = lambda gate: gate.get_gates(values=None, is_train=False).flatten().tolist()
        enc_self_attn_gates = [f(layer.self_attn.gate) for layer in self.encoder.layers]
        dec_self_attn_gates = [f(layer.self_attn.gate) for layer in self.decoder.layers]
        dec_enc_attn_gates = [f(layer.enc_attn.gate) for layer in self.decoder.layers]
        return enc_self_attn_gates, dec_self_attn_gates, dec_enc_attn_gates
    
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