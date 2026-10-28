"""Build five vector slides and insert them after page 8 of the reference PDF.

Run with A1/.venv/Scripts/python.exe. Requires PyMuPDF and reportlab.
The original PDF and AI_USAGE.md are never modified.
"""
from pathlib import Path
import sys
import json
import hashlib
import urllib.request
from fontTools.ttLib import TTFont as FontToolsFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / '.slides-build'
sys.path.insert(0, str(BUILD / 'deps'))
import fitz
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor

SOURCE = ROOT / 'Fashion-MNIST Comparison.pdf'
EXTRA = ROOT / 'Fashion-MNIST Model Deep Dives.pdf'
FINAL = ROOT / 'Fashion-MNIST Comparison - Expanded.pdf'
W, H = 1440, 810
NAVY, BLUE, CYAN, PALE = '#0C2D60', '#165DA5', '#19B7D5', '#E9F1F7'
BUILD.mkdir(exist_ok=True)
reference = fitz.open(SOURCE)
for family, url, weight in [('Heading', 'montserrat/Montserrat%5Bwght%5D.ttf', 700), ('Body', 'sourcesans3/SourceSans3%5Bwght%5D.ttf', 400)]:
    full = BUILD / (family + '-full.ttf')
    if not full.exists():
        urllib.request.urlretrieve('https://raw.githubusercontent.com/google/fonts/main/ofl/' + url, full)
    file = BUILD / (family + '.ttf')
    instantiateVariableFont(FontToolsFont(full), {'wght': weight}, inplace=True).save(file)
    pdfmetrics.registerFont(TTFont(family, str(file)))

c = canvas.Canvas(str(EXTRA), pagesize=(W, H))
c.setTitle('Fashion-MNIST: five model deep dives')
c.setAuthor('Team G-09')

def rect(x,y,w,h,color):
    c.setFillColor(HexColor(color)); c.rect(x,H-y-h,w,h,stroke=0,fill=1)

def text(s,x,y,w,size=26,font='Body',color=NAVY,maxh=160):
    style=ParagraphStyle('p',fontName=font,fontSize=size,leading=size*1.22,textColor=HexColor(color))
    p=Paragraph(s,style); _,h=p.wrap(w,maxh)
    assert h <= maxh, (s,h,maxh)
    p.drawOn(c,x,H-y-h)
    return h

def start(title, subtitle, number):
    rect(0,0,W,H,'#FFFFFF')
    text(title,90,57,1260,44,'Heading',maxh=65)
    rect(90,135,70,5,CYAN)
    text(subtitle,90,158,1260,27,maxh=45)
    text(str(number),1300,752,50,18,color=BLUE,maxh=25)

def flow(nodes):
    gap=32; width=(1260-gap*(len(nodes)-1))/len(nodes)
    for i,(label,detail) in enumerate(nodes):
        x=90+i*(width+gap)
        rect(x,230,width,132,PALE if i<len(nodes)-1 else BLUE)
        ink=NAVY if i<len(nodes)-1 else '#FFFFFF'
        text(label,x+16,246,width-32,23,'Heading',ink,maxh=60)
        text(detail,x+16,307,width-32,21,'Body',ink,maxh=52)
        if i<len(nodes)-1:
            c.setStrokeColor(HexColor(BLUE)); c.setLineWidth(2)
            mid=x+width+gap/2
            c.line(x+width+5,H-296,x+width+gap-5,H-296)
            c.line(mid+7,H-296,mid+1,H-290); c.line(mid+7,H-296,mid+1,H-302)

def col(x,title,body):
    text(title,x,407,590,27,'Heading',maxh=40)
    text(body,x,456,590,26,maxh=225)

def finish(model):
    runs=ROOT.parents[1]/'runs'
    group='m1' if model in ('linear','mlp') else 'm2'
    summary=json.loads((runs/group/model/'summary.json').read_text())
    test=json.loads((runs/group/model/'test_summary.json').read_text())
    # The comparison exporter names scores consistently across saved run schemas.
    comparison=json.loads((runs/'m2/comparison/comparison.json').read_text())['results']
    row=next(r for r in comparison if r['model']==model)
    assert row['parameters']==summary['parameters']
    rect(90,697,1260,48,PALE)
    metrics=[f"{row['parameters']:,} parameters",f"Accuracy {row['test_accuracy']*100:.2f}%",f"Macro-F1 {row['test_macro_f1']:.4f}",f"Selected epoch {row['selected_epoch']}"]
    for i,label in enumerate(metrics):
        text(label,108+i*310,707,300,23,maxh=32)
    text('Shapes describe one image. Scores and epochs: original PDF / comparison.json, seed 42.',90,760,1180,17,maxh=25)
    c.showPage()

start('Linear: one weighted score per class','The simplest baseline connects all 784 pixels directly to ten output scores.',9)
flow([('Flat pixels','784 values'),('Class weights','10 sets of 784'),('Weighted sum','z = Wx + b'),('Prediction','10 logits, argmax')])
col(90,'How one score is calculated','For each class, multiply every pixel by its learned weight, add the products, then add a bias.<br/><br/>W has shape 10 × 784. The bias has 10 values: 7,840 + 10 = 7,850 parameters.')
col(760,'What this baseline can learn','Each class learns a position-specific pixel pattern. There is no hidden layer or nonlinear feature extractor.<br/><br/>All models train on raw logits with cross-entropy. Softmax is unnecessary before the loss or argmax.')
finish('linear')

start('MLP: nonlinear combinations of pixels','Two hidden layers combine the flat input into increasingly abstract features.',10)
flow([('Flat pixels','784 values'),('Hidden layer 1','256 units'),('Hidden layer 2','128 units'),('Class scores','10 logits')])
col(90,'Inside each hidden layer','A dense layer mixes all its inputs. ReLU keeps positive activations and sets negative ones to zero.<br/><br/>Dropout 0.2 follows each ReLU during training. Validation and testing disable dropout.')
col(760,'What changes from Linear','Nonlinear hidden layers can combine pixel evidence in ways a single linear map cannot.<br/><br/>The MLP keeps every pixel, but uses no shared local filters. Its 4.24-point accuracy gain also comes with greater capacity.')
finish('mlp')

start('CNN: shared filters learn local patterns','Convolution keeps the image grid, while pooling reduces its spatial size.',11)
flow([('Image grid','1 × 28 × 28'),('Conv + pool','32 × 14 × 14'),('Conv + pool','64 × 7 × 7'),('Flatten + dense','3,136 to 128'),('Class scores','10 logits')])
col(90,'Inside the two convolution blocks','Each block uses a 3 × 3 convolution, padding 1, ReLU, then 2 × 2 max pooling.<br/><br/>Convolution preserves spatial size. Pooling halves it: 28 to 14 to 7. Shared filters reuse a detector at different locations.')
col(760,'The classifier holds most weights','Flatten 64 × 7 × 7 features, then dense 128, ReLU, dropout 0.2, and dense 10.<br/><br/>The first dense layer has 401,536 parameters, about 95% of the total. The best score here does not isolate a pure convolution effect.')
finish('cnn')

start('LSTM: a memory updated one row at a time','The 28 image rows form a spatial sequence, read from top to bottom.',12)
flow([('Row sequence','28 steps × 28 pixels'),('LSTM layer','128 hidden units'),('Final hidden state','128 values, h at row 28'),('Readout','Dropout 0.2 + dense 10')])
col(90,'How the memory changes','At each row, the forget gate controls old memory, the input gate controls new content, and the output gate exposes the hidden state.<br/><br/>Hidden and cell states start at zero for each image. The same LSTM weights process every row.')
col(760,'Exactly which state is classified','A single forward layer reads all 28 rows. Only its final hidden state enters the classifier.<br/><br/>The cell state carries memory internally. There is no averaging of row outputs. The 88.48% score is only two test images behind MLP.')
finish('lstm')

start('Transformer: patches exchange information','Self-attention connects all 49 patches, using learned positions to retain layout.',13)
flow([('Patch sequence','49 patches × 16 pixels'),('Embed + position','49 tokens × 64'),('2 encoder blocks','4 heads per block'),('Normalize + mean','49 tokens become 64'),('Readout','Dropout + dense 10')])
col(90,'Inside each encoder block','Each attention head compares queries and keys, then mixes values from all patches. Each head uses 16 features.<br/><br/>Pre-norm residual sublayers wrap attention and a 64-to-128-to-64 feedforward network with GELU. Dropout is 0.2.')
col(760,'From an image to one prediction','Split the image into a 7 × 7 grid of 4 × 4 patches in raster order. Project 16 pixels to 64 features, add positions, then apply dropout.<br/><br/>After the blocks, layer-normalize and average all tokens. No CLS token or causal mask. Training starts from scratch.')
finish('transformer')
c.save()

extra=fitz.open(EXTRA)
merged=fitz.open()
merged.insert_pdf(reference,from_page=0,to_page=7)
merged.insert_pdf(extra)
merged.insert_pdf(reference,from_page=8)
merged.set_metadata({'title':'Fashion-MNIST Comparison - Expanded','author':'Team G-09'})
merged.save(FINAL,garbage=4,deflate=True)
assert len(merged)==20 and len(extra)==5
for i,p in enumerate(extra):
    p.get_pixmap(matrix=fitz.Matrix(1,1)).save(BUILD/f'new-{i+9}.png')
# Confirm insertion preserved every original page's text and rendered appearance.
for i,p in enumerate(reference):
    q=merged[i if i<8 else i+5]
    assert p.get_text()==q.get_text()
    assert p.get_pixmap(matrix=fitz.Matrix(.3,.3)).samples==q.get_pixmap(matrix=fitz.Matrix(.3,.3)).samples
(BUILD/'validation.json').write_text(json.dumps({'total_slides':20,'new_slides':[9,10,11,12,13], 'original_pages_preserved':True,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()},indent=2))
print(FINAL)
print(EXTRA)
