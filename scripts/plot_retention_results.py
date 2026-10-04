"""Plot separately labelled retrieval populations from measured, frozen reports."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def run():
    folder=ROOT/'reports/retention_v2';report=json.loads((folder/'test_evaluation/summary.json').read_text(encoding='utf8'))
    historical=json.loads((folder/'historical_government/summary.json').read_text(encoding='utf8'))
    groups=[('InsuranceQA: historical test, n=2,000\n27,413 answer candidates',report['summaries']['historical_insuranceqa']['all']),
            ('Government: new test, n=81 / 14 sources\n47,969 answer candidates',report['summaries']['fresh_multidomain']['government']),
            ('General paragraphs: new test, n=600 / 24 titles\n47,969 answer candidates',report['summaries']['fresh_multidomain']['general']),
            ('Government: historical regression, n=416\n27,829 answer candidates',historical['summaries'])]
    labels=['BGE','Public reranker','Previous trained','Selected update'];keys=['bge','public','previous','selected']
    colors=['#9ca3af','#d6a548','#7ea9c4','#16557d'];fig,axes=plt.subplots(2,2,figsize=(12.5,8.5))
    for ax,(title,arms) in zip(axes.flat,groups):
        values=[100*arms[k]['hit_at_10'] for k in keys];bars=ax.bar(np.arange(4),values,color=colors,width=.65)
        ax.bar_label(bars,labels=[f'{x:.2f}%' for x in values],padding=4,fontsize=10)
        ax.set_xticks(np.arange(4),labels,fontsize=9);ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Hit@10 (%)')
        ax.set_title(title,fontsize=11,pad=12);ax.spines[['top','right']].set_visible(False);ax.yaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Mixed-domain reranker training: separate evaluation populations',fontsize=15,y=.99)
    fig.text(.5,.015,'BGE remains frozen. Retrieval metrics only; not generation accuracy. Source-cluster paired intervals are in the report.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96));out=ROOT/'docs/assets/retention_results.png';fig.savefig(out,dpi=180);plt.close(fig);print(out)

if __name__=='__main__':run()
