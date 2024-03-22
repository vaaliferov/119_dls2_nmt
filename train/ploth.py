import plotly
import plotly.subplots

def html_plot_loss_and_bleu(loss, bleu, path):
    fig = plotly.subplots.make_subplots(specs=[[{'secondary_y': True}]])
    lp = {'name':'loss', 'line_color':'#1f77b4', 'opacity':1.00}
    bp = {'name':'bleu', 'line_color':'#ff7f0e', 'opacity':0.35}
    loss_scatter = plotly.graph_objects.Scatter(y=loss, **lp)
    bleu_scatter = plotly.graph_objects.Scatter(y=bleu, **bp)
    fig.add_trace(loss_scatter, secondary_y=False)
    fig.add_trace(bleu_scatter, secondary_y=True)
    fig.update_xaxes(title_text='iterations')
    fig.update_yaxes(title_text='loss', secondary_y=False)
    fig.update_yaxes(title_text='bleu', secondary_y=True)
    html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    with open(path, 'w') as fd: fd.write(html)