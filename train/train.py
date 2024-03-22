import os
import re
import sys
import torch
import torchtext

import numpy as np
from time import time
import youtokentome as yttm
from datetime import timedelta

from tg import *
from plot import *
from utils import *
from config import *
from secret import *
from schedule import *

from model import Model
from loader import DataLoader

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tg = TG(TG_NOTIFY_BOT_TOKEN, TG_BOT_OWNER_ID, local=True)

def log(text):
    fd = open(LOG_FILE_PATH, 'a+')
    fd.write(f'{text}\n')
    tg.send_message(text)
    fd.close()

excluded = {'data', 'pics', 'telegram'}
zip_dir('.', excluded, SOURCES_ZIP_FILE_PATH)
tg.send_document(SOURCES_ZIP_FILE_PATH)

src_tok = yttm.BPE(SRC_TOKENIZER_PATH)
trg_tok = yttm.BPE(TRG_TOKENIZER_PATH)
src_vocab_size = len(src_tok.vocab())
trg_vocab_size = len(trg_tok.vocab())
log(f'vocab: {src_vocab_size}, {trg_vocab_size}')

model = Model(src_vocab_size, trg_vocab_size, device)
log(f'model parameters: {model.count_parameters()}')

if os.path.isfile(MODEL_PATH):
    log('loading model weights...')
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=False)

with open(SRC_SAMP_FILE_PATH) as fd:
    log('tokenizing source sentences...')
    u16 = lambda lst: np.array(lst, dtype='uint16')
    src_data = [u16(src_tok.encode(l[:-1])) for l in fd]

with open(TRG_SAMP_FILE_PATH) as fd:
    log('tokenizing target sentences...')
    u16 = lambda lst: np.array(lst, dtype='uint16')
    trg_data = [u16(trg_tok.encode(l[:-1])) for l in fd]

data = list(zip(src_data, trg_data))
data = np.array(data, dtype=object)

log('splitting data...')
np.random.default_rng(SEED).shuffle(data)
sizes = [int(r * len(data)) for r in SPLIT_RATIO]
train, valid, test = np.split(data, np.cumsum(sizes)[:-1])
log(f'data: {len(train)}, {len(valid)}, {len(test)}')

log('building loaders...')
chunk_size = None if PRUNING else NUM_BATCHES_IN_CHUNK
train_loader = DataLoader(train, BATCH_SIZE, chunk_size, shuffle=True)
valid_loader = DataLoader(valid, BATCH_SIZE, chunk_size, shuffle=False)
test_loader = DataLoader(test, BATCH_SIZE, chunk_size, shuffle=False)
log(f'batches: {len(train_loader)}, {len(valid_loader)}, {len(test_loader)}')

log('setting up the optimizer and loss function...')
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

warmup_steps = 0 if PRUNING else WARMUP_STEPS
training_steps = len(train_loader) * NUM_EPOCHS
lr_schedule = cosine_learning_rate(LEARNING_RATE, warmup_steps, training_steps)
start, stop = (CUR_EPOCH - 1) * len(train_loader), CUR_EPOCH * len(train_loader)
optimizer = ScheduledAdam(model.parameters(), lr_schedule[start:stop])
plot_learning_rate(lr_schedule, start, stop, LEARNING_RATE_PLOT_PATH)
tg.send_document(LEARNING_RATE_PLOT_PATH)

def bleu(refs, hyps):
    p = {'max_n': 2, 'weights': (0.5, 0.5)}
    hyps = [re.findall('\w+', s.lower()) for s in hyps]
    refs = [[re.findall('\w+', s.lower())] for s in refs]
    return torchtext.data.metrics.bleu_score(hyps, refs, **p)

def nn_epoch(model, optimizer, criterion, loader, device, is_train):

    start_ts = time()
    gates_last_state = None
    last_batch_log_ts = time()
    last_checkpoint_ts = time()
    
    loss_history = np.zeros(len(loader))
    bleu_history = np.zeros(len(loader))

    torch.set_grad_enabled(is_train)
    model.train() if is_train else model.eval()
    
    for i, batch in enumerate(loader):
        
        start_batch_ts = time()
        sources = batch[0].to(device)
        targets = batch[1].to(device)
        
        if is_train: optimizer.zero_grad()
        outputs, _, _, _ = model(sources, targets[:,:-1])
        l_targets = targets[:,1:].contiguous().view(-1)
        l_outputs = outputs.view(-1, outputs.shape[-1])

        batch_loss = criterion(l_outputs, l_targets)
        if PRUNING: batch_loss += model.get_penalty()
        
        if is_train:
            batch_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_
            norm(model.parameters(), GRAD_CLIP)
            optimizer.step()

        m_sources = src_tok.decode(sources.tolist(), ignore_ids=SPECIAL_IDS)
        m_targets = trg_tok.decode(targets.tolist(), ignore_ids=SPECIAL_IDS)
        m_outputs = trg_tok.decode(outputs.argmax(-1).tolist(), ignore_ids=SPECIAL_IDS)
        
        loss_history[i] = batch_loss.item()
        bleu_history[i] = bleu(m_targets, m_outputs)
        batch_time = timedelta(seconds=(time() - start_batch_ts))

        if (time() - last_batch_log_ts) > BATCH_LOG_PERIOD:

            log((f"{get_gpu_info()}, "
                 f"batch: {i+1} / {len(loader)}, time: {batch_time}, "
                 f"loss: {loss_history[i]:.4f}, bleu: {bleu_history[i]:.4f}"))
            
            last_batch_log_ts = time()
            s = np.random.choice(np.arange(len(m_targets)))
            log(m_sources[s]); log(m_targets[s]); log(m_outputs[s])

            if i > LOSS_BLEU_WINDOW_SIZE:
                s = slice(i - LOSS_BLEU_WINDOW_SIZE + 1, i + 1)
                loss_slice, bleu_slice = loss_history[s], bleu_history[s]
                plot_loss_and_bleu(loss_slice, bleu_slice, LOSS_BLEU_PLOT_PATH)
                tg.send_document(LOSS_BLEU_PLOT_PATH)
            
            if PRUNING and is_train:
                if gates_last_state != None:
                    plot_gates(model.get_gates(), gates_last_state, ATTN_GATES_PLOT_PATH)
                    tg.send_document(ATTN_GATES_PLOT_PATH)
                gates_last_state = model.get_gates()

        if (time() - last_checkpoint_ts) > CHECKPOINT_PERIOD:
            loss_slice, bleu_slice = loss_history[:i+1], bleu_history[:i+1]
            plot_loss_and_bleu(loss_slice, bleu_slice, LOSS_BLEU_PLOT_PATH)
            torch.save(model.state_dict(), MODEL_PATH)
            np.save(LOSS_HISTORY_PATH, loss_slice)
            np.save(BLEU_HISTORY_PATH, bleu_slice)
            tg.send_document(LOSS_BLEU_PLOT_PATH)
            tg.send_document(LOSS_HISTORY_PATH)
            tg.send_document(BLEU_HISTORY_PATH)
            tg.send_document(MODEL_PATH)
            last_checkpoint_ts = time()
    
    epoch_time = timedelta(seconds=(time() - start_ts))
    return {'loss': loss_history, 'bleu': bleu_history, 'time': epoch_time}

log('training...')
t = nn_epoch(model, optimizer, criterion, train_loader, device, is_train=True)
log(f"[train] time: {t['time']}, loss: {t['loss'].mean():.4f}, bleu: {t['bleu'].mean():.4f}")
plot_loss_and_bleu(t['loss'], t['bleu'], LOSS_BLEU_PLOT_PATH)
tg.send_document(LOSS_BLEU_PLOT_PATH)

log('validating...')
v = nn_epoch(model, None, criterion, valid_loader, device, is_train=False)
log(f"[valid] time: {v['time']}, loss: {v['loss'].mean():.4f}, bleu: {v['bleu'].mean():.4f}")
plot_loss_and_bleu(v['loss'], v['bleu'], LOSS_BLEU_PLOT_PATH)
tg.send_document(LOSS_BLEU_PLOT_PATH)

log('saving...')
torch.save(model.state_dict(), MODEL_PATH)
np.save(TRAIN_LOSS_HISTORY_PATH, t['loss'])
np.save(TRAIN_BLEU_HISTORY_PATH, t['bleu'])
np.save(VALID_LOSS_HISTORY_PATH, v['loss'])
np.save(VALID_BLEU_HISTORY_PATH, v['bleu'])

log('uploading...')
tg.send_document(MODEL_PATH) # log(file_id)
tg.send_document(LOG_FILE_PATH)
tg.send_document(SRC_TOKENIZER_PATH)
tg.send_document(TRG_TOKENIZER_PATH)
tg.send_document(TRAIN_LOSS_HISTORY_PATH)
tg.send_document(TRAIN_BLEU_HISTORY_PATH)
tg.send_document(VALID_LOSS_HISTORY_PATH)
tg.send_document(VALID_BLEU_HISTORY_PATH)
log('done!')