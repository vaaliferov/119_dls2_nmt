import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.use('Agg')

def save_and_close(fig, path):
    sp = {
        'dpi': 90, 
        'edgecolor': 'none', 
        'facecolor': 'white', 
        'bbox_inches': 'tight', 
        'transparent': False
    }
    fig.savefig(path, **sp)
    plt.close(fig)
    
def set_locator_and_formatter(axis, labels):
    indices = range(len(labels))
    loc = ticker.FixedLocator(indices)
    fmt = ticker.FixedFormatter(labels)
    axis.set_major_locator(loc)
    axis.set_major_formatter(fmt)

def plot_learning_rate(data, start, stop, path):
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    ax.set_ylabel('learning rate')
    ax.plot(data, color='tab:blue')
    x = np.arange(start, stop)
    p = {'color':'tab:green','alpha':0.1}
    ax.fill_between(x, max(data) * 1.02, **p)
    ax.set_ylim(0, max(data) * 1.02)
    save_and_close(fig, path)

def plot_loss_and_bleu(loss, bleu, path):
    fig, ax_loss = plt.subplots()
    fig.set_size_inches(10,6)
    ax_bleu = ax_loss.twinx()
    lc, bc = 'tab:blue', 'tab:orange'
    ax_loss.set_ylabel('loss', color=lc)
    ax_bleu.set_ylabel('bleu', color=bc)
    ax_loss.tick_params(axis='y', labelcolor=lc)
    ax_bleu.tick_params(axis='y', labelcolor=bc)
    ax_loss.plot(loss, label='loss', color=lc)
    ax_bleu.plot(bleu, label='bleu', color=bc, alpha=0.35)
    save_and_close(fig, path)

def plot_toks_num_hist(data, path):
    fig, ax = plt.subplots()
    ax.set_ylabel('number of sentences')
    ax.set_xlabel('number of tokens in sentence')
    ax.hist(data, bins=30, log=True, ec='steelblue')
    save_and_close(fig, path)

def plot_attn(attn, xlabels, ylabels, path):
    fig, axes = plt.subplots(6, 8, figsize=(24,14))
    attn = [ha for la in attn for ha in la[0]]
    for ax, at in zip(axes.flat, attn):
        ax.matshow(at, cmap='gray')
        set_locator_and_formatter(ax.xaxis, xlabels)
        set_locator_and_formatter(ax.yaxis, ylabels)
        ax.set_xticklabels(xlabels, rotation=90, fontsize=6)
        ax.set_yticklabels(ylabels, fontsize=6)
    save_and_close(fig, path)

def plot_gates(a, b, path):
    d = (np.array(a) - np.array(b)).tolist()
    fig, axes = plt.subplots(3, 1, figsize=(5,11))
    for i, (ax, gs) in enumerate(zip(axes.flat, a)):
        ax.matshow(gs, cmap='gray', vmin=0, vmax=1)
        ax.set_ylabel(['enc_self','dec_self','dec_enc'][i]+'_attn')
        for (l, h), z in np.ndenumerate(gs):
            a, c = 'center', lambda z: ['0','1'][int(z <= 0.5)]
            p = {'fontsize': 7, 'ha': a, 'va': a, 'color':c(z)}
            t = f'{z:.2f}↓' if d[i][l][h] < 0 else f'{z:.2f}↑'
            ax.text(h, l, t, **p)
    save_and_close(fig, path)