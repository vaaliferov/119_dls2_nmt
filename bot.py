import torch
import telegram
import telegram.ext
import youtokentome as yttm

from plot import *
from config import *
from secret import *
from model import Model

src_tok = yttm.BPE(SRC_TOKENIZER_PATH)
trg_tok = yttm.BPE(TRG_TOKENIZER_PATH)
src_vocab_size = len(src_tok.vocab())
trg_vocab_size = len(trg_tok.vocab())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(src_vocab_size, trg_vocab_size, device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state, strict=True)

def send_document(context, chat_id, file_path):
    with open(file_path, 'rb') as fd: 
        context.bot.send_document(chat_id, fd)

def handle_text(update, context):
    text = update.message.text
    user = update.message.from_user
    chat_id = update.message.chat_id
    attn_maps, beam_search = False, False
    
    if user['id'] != TG_BOT_OWNER_ID:
        msg = f"@{user['username']} {user['id']}"
        context.bot.send_message(TG_BOT_OWNER_ID, msg)
        context.bot.send_message(TG_BOT_OWNER_ID, text)
    
    if text == '/start':
        usage = 'Please, send me a text in Russian'
        context.bot.send_message(chat_id, usage)
        return None

    if text[-1] == '*':
        text = text[:-1]
        beam_search = True

    if text[-1] == '#':
        text = text[:-1]
        attn_maps = True

    text = text[0].upper() + text[1:]
    if text[-1] not in '.!?': text += '.'

    src = src_tok.encode(text, bos=True, eos=True)
    trg, enc_self_attn, dec_self_attn, dec_enc_attn = model.greedy_generate(src)
    result = trg_tok.decode(trg, ignore_ids=SPECIAL_IDS)
    context.bot.send_message(chat_id, result[0])

    if beam_search and len(src) < 20:
        beam_trg, _, _, _ = model.beam_generate(src)
        beam_result = trg_tok.decode(beam_trg, ignore_ids=SPECIAL_IDS)
        beam_result = [f'{i+1}. {r}' for i, r in enumerate(beam_result)]
        context.bot.send_message(chat_id, '\n'.join(beam_result))
    
    if attn_maps and len(src) < 20 and len(trg) < 20:
        context.bot.send_message(chat_id, 'please wait.. (~15s)')
        src_labels = [src_tok.id_to_subword(t) for t in src]
        trg_labels = [trg_tok.id_to_subword(t) for t in trg]
        
        plot_attn(enc_self_attn, src_labels, src_labels, ENC_SELF_ATTN_PLOT_PATH)
        plot_attn(dec_enc_attn, src_labels, trg_labels[1:], DEC_ENC_ATTN_PLOT_PATH)
        plot_attn(dec_self_attn, trg_labels[1:], trg_labels[1:], DEC_SELF_ATTN_PLOT_PATH)
        
        send_document(context, chat_id, ENC_SELF_ATTN_PLOT_PATH)
        send_document(context, chat_id, DEC_SELF_ATTN_PLOT_PATH)
        send_document(context, chat_id, DEC_ENC_ATTN_PLOT_PATH)

f = telegram.ext.Filters.text
h = telegram.ext.MessageHandler
u = telegram.ext.Updater(TG_NMT_BOT_TOKEN)
u.dispatcher.add_handler(h(f,handle_text))
u.start_polling(); u.idle()