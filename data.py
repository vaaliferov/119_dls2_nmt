import os
import re
import sys
import random
import shutil
import zipfile
import hashlib
import requests

import numpy as np
from tqdm import tqdm
import youtokentome as yttm

from tg import *
from plot import *
from utils import *
from secret import *
from config import *

cmds = ('fetch','sample','train','check','upload')
if len(sys.argv) != 2 or sys.argv[1] not in cmds:
    print(f"usage: {sys.argv[0]} [{'|'.join(cmds)}]")
    sys.exit(1)

random.seed(SEED)
np.random.seed(SEED)

tg = TG(TG_NOTIFY_BOT_TOKEN, TG_BOT_OWNER_ID, local=True)

def log(text):
    fd = open(LOG_FILE_PATH, 'a+')
    fd.write(f'{text}\n')
    tg.send_message(text)
    print(text)
    fd.close()

def download_file(url, path):
    rsp = requests.get(url, stream=True)
    t = int(rsp.headers.get('content-length', 0))
    tq = tqdm(total=t, unit='iB', unit_scale=True)
    with open(path, 'wb') as fd:
        for data in rsp.iter_content(1024**2):
            tq.update(len(data)); fd.write(data)
    tq.close()

def clean_line(line, lang):
    if line[0] == '-': line = line[1:]
    a = 'A-Za-z' if lang == 'en' else 'ЁёА-Яа-я'
    line = re.sub(f"[^-'{a}0-9 ,.!?\n]", '', line)
    return line.strip(' ')

def mostly_in_russian(s):
    s = [c for c in s.lower() if c.isalpha()]
    c = sum(['а' <= c <= 'я' for c in s])
    return c / (len(s) + 1e-10) >= 0.5

def mostly_in_english(s):
    s = [c for c in s.lower() if c.isalpha()]
    c = sum(['a' <= c <= 'z' for c in s])
    return c / (len(s) + 1e-10) >= 0.5

def have_similar_lens(s1, s2):
    lmin = min(len(s1), len(s2))
    lmax = max(len(s1), len(s2))
    return lmin / (lmax + 1e-10) > 0.3

def good_pair(s, t):
    return (
        len(s) >= MIN_CHAR_NUM and 
        len(s) <= MAX_CHAR_NUM and 
        len(t) >= MIN_CHAR_NUM and 
        len(t) <= MAX_CHAR_NUM and  
        mostly_in_russian(s) and 
        mostly_in_english(t) and 
        have_similar_lens(s, t)
    )

if sys.argv[1] == 'fetch':
    
    log(f'downloading data..')
    os.makedirs(DATA_PATH, exist_ok=True)
    download_file(ZIP_FILE_URL, ZIP_FILE_PATH)

    log('extracting files..')
    zf = zipfile.ZipFile(ZIP_FILE_PATH)
    zf.extract(SRC_FILE_NAME, DATA_PATH)
    zf.extract(TRG_FILE_NAME, DATA_PATH)
    shutil.move(SRC_FILE_PATH, SRC_RAW_FILE_PATH)
    shutil.move(TRG_FILE_PATH, TRG_RAW_FILE_PATH)
    zf.close()

if sys.argv[1] == 'sample':

    num_lines = 0
    log('counting lines..')
    with open(SRC_RAW_FILE_PATH) as fd:
        for _ in fd: num_lines += 1
    log(f'lines: {num_lines}')
    
    log('sampling indices..')
    idx = np.arange(num_lines)
    np.random.default_rng(SEED).shuffle(idx)
    samp = set(idx[:SAMPLE_SIZE])
    log(f'sample size: {len(samp)}')
    log(f'first index: {idx[0]}')

    log('extracting pairs..')
    sm, tm, unique_pairs = 0, 0, set()
    sof = open(SRC_RAW_FILE_PATH, 'r')
    tof = open(TRG_RAW_FILE_PATH, 'r')
    ssf = open(SRC_SAMP_FILE_PATH, 'w')
    tsf = open(TRG_SAMP_FILE_PATH, 'w')
    tq = tqdm(zip(sof, tof), total=num_lines)

    for i, (s, t) in enumerate(tq):
        if i in samp:
            s = clean_line(s, 'ru')
            t = clean_line(t, 'en')
            if good_pair(s, t):
                e = (s + t).encode('utf-8')
                h = hashlib.md5(e).hexdigest()
                if h not in unique_pairs:
                    if len(s) > sm: sm = len(s)
                    if len(t) > tm: tm = len(t)
                    ssf.write(s); tsf.write(t)
                    unique_pairs.add(h)

    for fd in (sof, tof, ssf, tsf): fd.close()
    log(f'unique pairs: {len(unique_pairs)}')
    log(f'max sent lens: {sm}, {tm}')

if sys.argv[1] == 'train':

    log('training source tokenizer..')
    yttm.BPE.train(
        data=SRC_SAMP_FILE_PATH, 
        vocab_size=SRC_VOCAB_SIZE, 
        model=SRC_TOKENIZER_PATH
    )

    log('training target tokenizer..')
    yttm.BPE.train(
        data=TRG_SAMP_FILE_PATH, 
        vocab_size=TRG_VOCAB_SIZE, 
        model=TRG_TOKENIZER_PATH
    )

if sys.argv[1] == 'check':
    
    src_tok = yttm.BPE(SRC_TOKENIZER_PATH)
    trg_tok = yttm.BPE(TRG_TOKENIZER_PATH)
    
    with open(SRC_SAMP_FILE_PATH) as fd:
        log('tokenizing source sentences...')
        src_lens = [len(src_tok.encode(l[:-1])) for l in fd]

    with open(TRG_SAMP_FILE_PATH) as fd:
        log('tokenizing target sentences...')
        trg_lens = [len(trg_tok.encode(l[:-1])) for l in fd]
    
    log(f'min_toks_num: {np.min(src_lens)}, {np.min(trg_lens)}')
    log(f'max_toks_num: {np.max(src_lens)}, {np.max(trg_lens)}')
    log(f'avg_toks_num: {np.mean(src_lens):.1f}, {np.mean(trg_lens):.1f}')
    
    log('building histograms...')
    plot_toks_num_hist(src_lens, SRC_TOKS_NUM_HIST_PATH)
    plot_toks_num_hist(trg_lens, TRG_TOKS_NUM_HIST_PATH)
    tg.send_document(SRC_TOKS_NUM_HIST_PATH)
    tg.send_document(TRG_TOKS_NUM_HIST_PATH)

if sys.argv[1] == 'upload':

    log('compressing..')
    zf = zipfile.ZipFile(f'{DATA_PATH}/data.zip', mode='w')
    zf.write(SRC_SAMP_FILE_PATH, compress_type=zipfile.ZIP_DEFLATED)
    zf.write(TRG_SAMP_FILE_PATH, compress_type=zipfile.ZIP_DEFLATED)
    zf.write(SRC_TOKENIZER_PATH, compress_type=zipfile.ZIP_DEFLATED)
    zf.write(TRG_TOKENIZER_PATH, compress_type=zipfile.ZIP_DEFLATED)
    zf.close()

    log('uploading..')
    log(tg.send_document(f'{DATA_PATH}/data.zip'))