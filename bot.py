import asyncio, torch
import re, os, argparse
import youtokentome as yttm
from transformer import Model

from telegram import Update
from telegram.ext import filters, Application
from telegram.ext import MessageHandler, CommandHandler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int, help='bot owner id')
    parser.add_argument('token', type=str, help='bot token')
    return parser.parse_args()

args = parse_args()
src_tok = yttm.BPE('ru_bpe.yttm')
trg_tok = yttm.BPE('en_bpe.yttm')
state = torch.load('model.pt', map_location='cpu')


params = {
    
    'bos_idx': 2,
    'eos_idx': 3,
    'pad_idx': 0,
    
    'n_heads': 8,
    'n_layers': 6,
    'dropout': 0.1,
    'max_len': 100,
    'device': 'cpu',

    'pf_dim': 512,
    'hid_dim': 256,
    
    'inp_dim': len(src_tok.vocab()),
    'out_dim': len(trg_tok.vocab())
}

model = Model(**params)
model.load_state_dict(state, strict=True)


def preprocess(text):

    pat = '[^ЁёА-Яа-я0-9 ,.!?-]'
    text = re.sub(pat, '', text).strip()

    if len(text):
        text = text[0].upper() + text[1:]
        if text[-1] not in '.!?': text + '.'

    return text


def translate(text):
    src = src_tok.encode(text, bos=True, eos=True)
    trg, _, _, _ = model.greedy_generate(src)
    return trg_tok.decode(trg, ignore_ids={0,1,2,3})[0]


async def handle_start(update, context):
    usage = 'Please, type something in Russian'
    await update.message.reply_text(usage)


async def handle_text(update, context):

    user = update.message.from_user
    chat_id = update.message.chat_id
    text = preprocess(update.message.text)

    if user['id'] != args.id:
        msg = f"@{user['username']} {user['id']}"
        await context.bot.send_message(args.id, msg)
        await context.bot.send_message(args.id, text)

    if len(text) > 2:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, translate, text)
        await update.message.reply_text(result)

    else: await update.message.reply_text('text is too short')


app = Application.builder().token(args.token).build()
app.add_handler(CommandHandler('start', handle_start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
app.run_polling(allowed_updates=Update.ALL_TYPES)