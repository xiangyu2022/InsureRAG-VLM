"""A static research figure from frozen paired results, with no invented values."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def run():
    data=json.loads((ROOT/'reports/condition_listwise_v1/test_evaluation/summary.json').read_text(encoding='utf8'))
    specs=[('historical_insuranceqa','all','FAQ\n2,000 historical'),('historical_multidomain','general','General\n600 historical'),('fresh_government','all','Government\n157 fresh')]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12.8,4.6),gridspec_kw={'width_ratios':[1.25,1]},layout='constrained')
    colors=['#93a1b0','#c49b51','#64788d','#237f78'];names=['BGE','Public / matched','Previous / matched','Selected']
    x=np.arange(3);width=.19
    for j,(arm,label,color) in enumerate(zip(['bge','public_matched','previous_matched','selected'],names,colors)):
        vals=[100*data['summaries'][c][p][arm]['hit_at_10'] for c,p,_ in specs]
        bars=ax.bar(x+(j-1.5)*width,vals,width,label=label,color=color)
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    ax.set_xticks(x,[label for _,_,label in specs]);ax.set_ylim(0,109);ax.set_ylabel('Hit@10 (%)');ax.set_title('A  Retrieval performance',loc='left',weight='bold')
    ax.legend(loc='upper left',fontsize=8,frameon=False,ncols=2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    comparisons=[('historical_insuranceqa','all','FAQ / 2,000'),('historical_government','all','Government / 416'),
        ('historical_multidomain','government','Government / 81'),('historical_multidomain','general','General / 600'),('fresh_government','all','Fresh government / 157')]
    for i,(c,p,label) in enumerate(comparisons):
        row=data['paired_selected_minus_controls'][c][p]['previous_matched']
        delta=100*row.get('hit10_difference',row.get('hit_at_10_difference'))
        lo,hi=[100*v for v in row.get('label_cluster_bootstrap_95',row.get('source_cluster_bootstrap_95ci'))]
        bx.errorbar(delta,i,xerr=[[max(0,delta-lo)],[max(0,hi-delta)]],fmt='o',color='#237f78',capsize=4)
    bx.axvline(0,color='#777777',ls='--',lw=1);bx.set_yticks(range(len(comparisons)),[r[2] for r in comparisons]);bx.invert_yaxis()
    bx.set_xlabel('Selected minus matched previous (percentage points)');bx.set_title('B  Training effect with 95% cluster intervals',loc='left',weight='bold',fontsize=10)
    bx.grid(axis='x',alpha=.15)
    for panel in [ax,bx]:panel.spines[['top','right']].set_visible(False)
    fig.suptitle('InsureRAG: condition-oriented retrieval experiment',weight='bold')
    folder=ROOT/'docs/assets';folder.mkdir(exist_ok=True);fig.savefig(folder/'condition_results.png',dpi=180);fig.savefig(folder/'condition_results.svg');plt.close(fig)
if __name__=='__main__':run()
