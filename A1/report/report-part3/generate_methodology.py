"""Verify architecture facts and draw original methodology diagrams; no training."""
from pathlib import Path
import hashlib
import json
import sys
import inspect

HERE = Path(__file__).resolve().parent
A1 = HERE.parent
sys.path.insert(0, str(A1))
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from a1.models import build_model, count_parameters, input_mode
from a1.data import to_representation

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def box(ax, x, y, w, h, label, color='#edf3f7', size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',
                              facecolor=color,edgecolor='#345568',linewidth=1))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color='#142f3e')

def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,
                               color='#345568',linewidth=1.1))

def main():
    (HERE/'figures').mkdir(exist_ok=True)
    (HERE/'qa').mkdir(exist_ok=True)
    before = HERE/'qa'/'preservation_before.json'
    protected = [A1.parent/'AI_USAGE.md', HERE/'Report_Style_Template (11).pdf',
                 A1/'report'/'A1_M1.pdf', A1/'a1'/'models.py', A1/'a1'/'data.py',
                 A1/'a1'/'train.py', A1/'a1'/'evaluate.py']
    if not before.exists():
        before.write_text(json.dumps({str(p):sha(p) for p in protected},indent=2)+'\n')
    cache=json.loads((A1/'split_cache.json').read_text())
    records={}
    torch.manual_seed(42)
    image=torch.arange(784,dtype=torch.float32).reshape(1,28,28)
    assert torch.equal(to_representation(image,'flatten'),image.reshape(-1))
    assert torch.equal(to_representation(image,'row_seq'),image[0])
    patches=to_representation(image,'patch_seq')
    rebuilt=patches.reshape(7,7,4,4).permute(0,2,1,3).reshape(28,28)
    assert torch.equal(rebuilt,image[0])
    for name in ['linear','mlp','cnn','lstm','transformer']:
        model=build_model(name).eval()
        x=to_representation(image/784,input_mode(name)).unsqueeze(0).repeat(2,*([1]*len(to_representation(image,input_mode(name)).shape)))
        shapes=[]
        hooks=[]
        def hook(label):
            def capture(module,inputs,output):
                def shape(v):
                    return list(v.shape) if isinstance(v,torch.Tensor) else [shape(i) for i in v]
                shapes.append({'module':label,'output':shape(output)})
            return capture
        for label,module in model.named_modules():
            if label and not list(module.children()):
                hooks.append(module.register_forward_hook(hook(label)))
        with torch.inference_mode():
            output=model(x)
        for h in hooks: h.remove()
        assert list(output.shape)==[2,10] and torch.isfinite(output).all()
        run=A1/'runs'/('m1' if name in ['linear','mlp'] else 'm2')/name
        summary=json.loads((run/'summary.json').read_text())
        assert count_parameters(model)==summary['parameters']
        ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['state_dict'],strict=True)
        history=__import__('csv').DictReader((run/'history.csv').open())
        rows=list(history)
        assert len(rows)==10
        best=min(rows,key=lambda r:float(r['val_loss']))
        assert int(best['epoch'])==int(summary['best_epoch'])==ckpt['epoch']
        cfg=summary['config']
        assert all(cfg[k]==v for k,v in {'seed':42,'batch_size':128,'epochs':10,'learning_rate':0.001,'augment_train':False,'num_workers':0,'val_ratio':0.1}.items())
        assert summary['split_sha256']==cache['split_sha256']
        records[name]={'parameters':count_parameters(model),'input':list(x.shape),
                       'output':list(output.shape),'layers':shapes,'saved_training_device':summary['device'],
                       'config':cfg,'strict_checkpoint_load':True,'selection_matches_validation_loss':True,
                       'summary_sha256':sha(run/'summary.json'),'checkpoint_sha256':sha(run/'best.pt')}
    adam=torch.optim.Adam(build_model('linear').parameters(),lr=0.001)
    facts={'models':records,'normalization':{k:cache[k] for k in ['mean','std','seed','split_sha256','source_sha256']},
           'split_sizes':[len(cache['train_idx']),len(cache['val_idx']),10000],
           'adam_defaults':{k:v for k,v in adam.defaults.items() if isinstance(v,(int,float,bool,str,tuple,type(None)))},
           'representation_values_and_order_checked':True,'batch_forward_checked':True,
           'training_rerun':False,'torch_version':torch.__version__}
    (HERE/'qa'/'methodology_evidence.json').write_text(json.dumps(facts,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(10.4,3.7)); ax.set(xlim=(0,1.045),ylim=(0,1)); ax.axis('off')
    box(ax,.015,.77,.255,.18,'Official training set\n60,000 images')
    box(ax,.36,.77,.275,.18,'Seed-42 stratified split\n54,000 train / 6,000 val')
    box(ax,.725,.77,.255,.18,'Official test set\n10,000 images')
    arrow(ax,(.27,.86),(.36,.86))
    box(ax,.015,.43,.29,.22,'Fit mean and SD on training only\nReuse on all three partitions')
    box(ax,.38,.43,.29,.22,'Choose one representation\nvector / grid / rows / patches\nOne independent model per run')
    box(ax,.745,.43,.235,.22,'Ten class logits\nCross-entropy + Adam\nTraining images only')
    arrow(ax,(.48,.77),(.18,.65)); arrow(ax,(.305,.54),(.38,.54)); arrow(ax,(.67,.54),(.745,.54))
    box(ax,.19,.06,.32,.21,'Validate after each epoch\nKeep lowest validation loss\nNo gradient updates on validation')
    box(ax,.62,.06,.32,.21,'Freeze selected weights\nEvaluate official test set\nMetrics, errors and timing')
    arrow(ax,(.865,.43),(.49,.27)); arrow(ax,(.51,.165),(.62,.165))
    ax.plot([.98,1.015,1.015],[.86,.86,.165],color='#536471',linestyle='--',linewidth=1)
    arrow(ax,(1.015,.165),(.94,.165))
    ax.text(1.027,.54,'Held out until evaluation',rotation=90,ha='center',va='center',fontsize=8,color='#536471')
    fig.savefig(HERE/'figures'/'methodology_pipeline.pdf',bbox_inches='tight',pad_inches=.05)
    fig.savefig(HERE/'figures'/'methodology_pipeline.png',dpi=180,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,3.0)); ax.set(xlim=(0,1.03),ylim=(0,1)); ax.axis('off')
    rows=[('Linear','784 pixels','Linear 784 to 10','10 logits'),
          ('MLP','784 pixels','256 ReLU / dropout\n128 ReLU / dropout','Linear 128 to 10'),
          ('CNN','1 x 28 x 28','Conv32 / pool / Conv64 / pool\nFlatten / Dense128 / dropout','Linear 128 to 10'),
          ('LSTM','28 rows x 28 pixels','One LSTM layer, hidden 128\nFinal hidden state / dropout','Linear 128 to 10'),
          ('Transformer','49 patches x 16 pixels','Embed64 + positions / 2 blocks\nLayer norm / mean / dropout','Linear 64 to 10')]
    for i,(name,inp,middle,out) in enumerate(rows):
        y=.82-i*.19
        ax.text(.005,y+.052,name,va='center',fontsize=9.5,weight='bold')
        box(ax,.20,y,.20,.12,inp,size=9)
        box(ax,.445,y,.335,.12,middle,size=8.2)
        box(ax,.83,y,.165,.12,out,size=8.5)
        arrow(ax,(.40,y+.06),(.445,y+.06)); arrow(ax,(.78,y+.06),(.83,y+.06))
    fig.savefig(HERE/'figures'/'model_architectures.pdf',bbox_inches='tight',pad_inches=.05)
    fig.savefig(HERE/'figures'/'model_architectures.png',dpi=180,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)
    print(json.dumps({'parameters':{k:v['parameters'] for k,v in records.items()},'checks':'passed','training_rerun':False},indent=2))

if __name__=='__main__': main()
