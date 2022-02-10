PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

SPECIAL_IDS = {
    BOS_IDX, 
    EOS_IDX, 
    UNK_IDX, 
    PAD_IDX
}

N_HEADS = 8
N_LAYERS = 6
PF_DIM = 512
HID_DIM = 256
DROPOUT = 0.1

SEED = 42
GRAD_CLIP = 1
CUR_EPOCH = 1
NUM_EPOCHS = 10
BATCH_SIZE = 128
MIN_CHAR_NUM = 2
MAX_CHAR_NUM = 140
MAX_TOKS_NUM = 100
WARMUP_STEPS = 70000
LEARNING_RATE = 0.0005

SAMPLE_SIZE = 12000000
SRC_VOCAB_SIZE = 30000
TRG_VOCAB_SIZE = 20000
BATCH_LOG_PERIOD = 300
CHECKPOINT_PERIOD = 1800
NUM_BATCHES_IN_CHUNK = 100
LOSS_BLEU_WINDOW_SIZE = 2000
SPLIT_RATIO = (0.95, 0.025, 0.025)

PRUNING = False
GATE_EPS = 1e-6
GATE_HARD = False
GATE_PENALTY = 0.05
GATE_TEMPERATURE = 0.5
GATE_LOCAL_REP = False
GATE_STRETCH_LIMITS = (-0.1, 1.1)

DATA_PATH = 'data'
MODEL_PATH = f'{DATA_PATH}/model.pt'
LOG_FILE_PATH = f'{DATA_PATH}/log.txt'
SRC_RAW_FILE_PATH = f'{DATA_PATH}/ru.txt'
TRG_RAW_FILE_PATH = f'{DATA_PATH}/en.txt'

SRC_SAMP_FILE_PATH = f'{DATA_PATH}/ru_samp.txt'
TRG_SAMP_FILE_PATH = f'{DATA_PATH}/en_samp.txt'
SRC_TOKENIZER_PATH = f'{DATA_PATH}/ru_bpe.yttm'
TRG_TOKENIZER_PATH = f'{DATA_PATH}/en_bpe.yttm'

URL_BASE = 'https://opus.nlpl.eu/download.php'
URL_QUERY = 'OpenSubtitles/v2018/moses/en-ru.txt.zip'
ZIP_FILE_URL = f'{URL_BASE}?f={URL_QUERY}'

SRC_FILE_NAME = 'OpenSubtitles.en-ru.ru'
TRG_FILE_NAME = 'OpenSubtitles.en-ru.en'
SRC_FILE_PATH = f'{DATA_PATH}/{SRC_FILE_NAME}'
TRG_FILE_PATH = f'{DATA_PATH}/{TRG_FILE_NAME}'
ZIP_FILE_PATH = f'{DATA_PATH}/en-ru.txt.zip'

ATTN_GATES_PLOT_PATH = f'{DATA_PATH}/attn_gates.png'
DEC_ENC_ATTN_PLOT_PATH = f'{DATA_PATH}/dec_enc_attn.png'
ENC_SELF_ATTN_PLOT_PATH = f'{DATA_PATH}/enc_self_attn.png'
DEC_SELF_ATTN_PLOT_PATH = f'{DATA_PATH}/dec_self_attn.png'
SRC_TOKS_NUM_HIST_PATH = f'{DATA_PATH}/src_toks_num_hist.png'
TRG_TOKS_NUM_HIST_PATH = f'{DATA_PATH}/trg_toks_num_hist.png'

LOSS_HISTORY_PATH = f'{DATA_PATH}/loss.npy'
BLEU_HISTORY_PATH = f'{DATA_PATH}/bleu.npy'
SOURCES_ZIP_FILE_PATH = f'{DATA_PATH}/sources.zip'
TRAIN_LOSS_HISTORY_PATH = f'{DATA_PATH}/train_loss.npy'
TRAIN_BLEU_HISTORY_PATH = f'{DATA_PATH}/train_bleu.npy'
VALID_LOSS_HISTORY_PATH = f'{DATA_PATH}/valid_loss.npy'
VALID_BLEU_HISTORY_PATH = f'{DATA_PATH}/valid_bleu.npy'

LOSS_BLEU_PLOT_PATH = f'{DATA_PATH}/loss_bleu.png'
LOSS_BLEU_HTML_PLOT_PATH = f'{DATA_PATH}/loss_bleu.html'
LEARNING_RATE_PLOT_PATH = f'{DATA_PATH}/learning_rate.png'
TRAIN_LOSS_BLEU_PLOT_PATH = f'{DATA_PATH}/train_loss_bleu.png'
VALID_LOSS_BLEU_PLOT_PATH = f'{DATA_PATH}/valid_loss_bleu.png'