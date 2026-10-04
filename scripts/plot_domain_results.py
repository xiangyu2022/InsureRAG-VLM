"""Plot the two distinct recorded retrieval tests without pooling their denominators."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs=[('Historical InsuranceQA\n2,000 questions / 27,413 answers',
           ROOT/'reports/domain_training_v2/test_evaluation/summary.json'),
          ('Government Q/A transfer\n416 questions / 37 sources / 27,829 answers',
           ROOT/'reports/hicric_government_qa_v1/transfer_v1/summary.json')]
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    colors=['#B9C9DB','#6281A3','#16766E']
    for ax,(title,path) in zip(axes,runs):
        data=json.loads(path.read_text(encoding='utf8'))['summaries']
        x=np.arange(2);width=.24
        for i,(arm,label) in enumerate([('bge','BGE'),('previous_untrained','Previous reranker'),('trained_selected','Domain-trained reranker')]):
            values=[100*data[arm][metric] for metric in ['hit_at_1','hit_at_10']]
            bars=ax.bar(x+(i-1)*width,values,width,color=colors[i],label=label,zorder=3)
            ax.bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=4,fontsize=8.5)
        ax.set_xticks(x,['Hit@1','Hit@10']);ax.set_ylim(0,108);ax.set_yticks([0,20,40,60,80,100])
        ax.set_ylabel('Questions with a labeled answer retrieved (%)');ax.set_title(title,fontsize=12,pad=14)
        ax.grid(axis='y',alpha=.18,zorder=0);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    fig.suptitle('InsureRAG: measured retrieval quality after domain training',fontsize=15)
    output=ROOT/'docs/assets/domain_training_results.png';output.parent.mkdir(exist_ok=True)
    fig.savefig(output,dpi=180);plt.close(fig)
    print(output)


if __name__=='__main__':main()
