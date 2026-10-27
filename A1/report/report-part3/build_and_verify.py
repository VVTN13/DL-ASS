"""Append the compiled Methodology to the retained PDF and verify deliverables."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import fitz
from PIL import Image, ImageDraw, ImageOps

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'Report_Style_Template (11).pdf'
OUTPUT=HERE/'report_with_methodology.pdf'
QA=HERE/'qa'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def normalized(text):
    return re.sub(r'\s+',' ',text).strip()

def main():
    source=fitz.open(SOURCE)
    new=fitz.open(HERE/'report.pdf')
    result=fitz.open()
    result.insert_pdf(source)
    result.insert_pdf(new)
    total=len(result)-1
    # Only contents additions and footer pagination change on the retained pages.
    # The supplied source PDF itself is never modified.
    for index,page in enumerate(result):
        if index==0: continue
        footer=fitz.Rect(475,804,545,824)
        page.add_redact_annot(footer,fill=(1,1,1))
        page.apply_redactions(images=0,graphics=0)
        text=f'Page {index}/{total}'
        width=fitz.get_text_length(text,fontname='cour',fontsize=7)
        page.insert_text((538.58-width,815.26),text,fontname='cour',fontsize=7)
    headings=[('3','Methodology'),('3.1','Overall System Architecture'),
              ('3.2','Baseline Approach'),('3.3','Proposed Models / Algorithms'),
              ('3.4','Mathematical Formulation'),('3.5','Training Strategy'),
              ('3.6','Evaluation Metrics')]
    toc=[[1,'Introduction',3],[2,'Background and Motivation',3],
         [2,'Problem Statement',3],[2,'Objectives and Scope',3],[2,'Proposed Approach',3],
         [1,'Dataset and Exploratory Analysis',4],[2,'Dataset Description',4],
         [2,'Data Preprocessing',4],[2,'Exploratory Data Analysis and Preprocessing',5],
         [2,'Preprocessing',10]]
    newtext=[p.get_text() for p in new]
    targets={}
    for number,title in headings:
        i=next(i for i,t in enumerate(newtext) if re.search(r'\n'+re.escape(number)+r'\s*\n'+re.escape(title),t))
        target=len(source)+i
        targets[number]=target
        toc.append([1 if number=='3' else 2,title,target+1])
    refs=next(i for i,t in enumerate(newtext) if re.search(r'\nReferences\s*\n',t))
    toc.append([1,'References for Methodology',len(source)+refs+1])
    page=result[1]
    for j,(number,title) in enumerate(headings+[('','References for Methodology')]):
        y=326+j*17
        page.insert_text((56.693 if number=='3' else 71.637,y),number,
                         fontname='tibo' if number=='3' else 'tiro',fontsize=10)
        page.insert_text((78 if number=='3' else 94.551,y),title,
                         fontname='tibo' if number=='3' else 'tiro',fontsize=10)
        page.insert_text((385,y),'. . . . . . . . . . . . . . . . . . . . . .',fontname='tiro',fontsize=10)
        target=targets[number] if number else len(source)+refs
        label=str(target)
        page.insert_text((538.58-fitz.get_text_length(label,fontname='tiro',fontsize=10),y),
                         label,fontname='tiro',fontsize=10)
        page.insert_link({'kind':fitz.LINK_GOTO,'from':fitz.Rect(70,y-11,540,y+3),'page':target})
    result.set_toc(toc)
    result.set_metadata({'title':'Assignment 1 - Retained Report with Section 3 Methodology',
                         'author':'Team G-09','subject':'Sections 1-3; original Sections 1-2 retained'})
    result.save(OUTPUT,garbage=4,deflate=True)
    result.close()
    doc=fitz.open(OUTPUT)
    log=(HERE/'report.log').read_text(encoding='utf-8',errors='replace')
    for issue in ['Overfull','Missing character','undefined references','Citation `']:
        assert issue not in log,issue
    text='\n'.join(p.get_text() for p in new)
    assert '??' not in text and '\ufffd' not in text
    assert not re.search(r'\b(TODO|VERIFY|TBD)\b',text)
    for number,title in headings:
        assert re.search(r'\n'+re.escape(number)+r'\s*\n'+re.escape(title),text)
    # Preserve the retained body text, including its original editorial choices.
    for i,old in enumerate(source):
        region=fitz.Rect(0,0,old.rect.width,790)
        old_text=normalized(old.get_text(clip=region))
        new_text=normalized(doc[i].get_text(clip=region))
        if i==1:
            assert new_text.startswith(old_text)
        else:
            assert old_text==new_text,i
    before=json.loads((QA/'preservation_before.json').read_text())
    for path,digest in before.items(): assert sha(Path(path))==digest,path
    for i,page in enumerate(new):
        assert len(page.get_text().strip())>850,(i+1,'sparse page')
        for block in page.get_text('dict')['blocks']:
            if block['type']==0:
                assert page.rect.contains(fitz.Rect(block['bbox'])),(i+1,block['bbox'])
        page.get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(QA/f'methodology-{i+1:02d}.png')
    for i in [0,1,len(source),len(doc)-1]:
        doc[i].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(QA/f'combined-{i+1:02d}.png')
    for start in range(0,len(new),4):
        sheet=Image.new('RGB',(1450,2040),'#dfe4e8')
        draw=ImageDraw.Draw(sheet)
        for j,i in enumerate(range(start,min(start+4,len(new)))):
            im=Image.open(QA/f'methodology-{i+1:02d}.png').convert('RGB')
            im=ImageOps.contain(im,(700,970))
            x=(j%2)*725+(725-im.width)//2
            y=(j//2)*1020+32
            sheet.paste(im,(x,y))
            draw.text((x,y-21),f'Methodology page {i+1}',fill='#243746')
        sheet.save(QA/f'contact-{start//4+1:02d}.png')
    report={'passed':True,'methodology_pages':len(new),'combined_pages':len(doc),
            'original_report_pages':len(source),'original_body_preserved':True,
            'original_files_unchanged':True,'AI_USAGE_unchanged':True,
            'resolved_citations_and_references':True,'all_new_pages_rendered':True,
            'no_overfull_boxes_or_missing_glyphs':True,'methodology_sha256':sha(HERE/'report.pdf'),
            'combined_sha256':sha(OUTPUT),'training_rerun':False,
            'new_sections':dict(headings),'visual_review':'See review_notes.md'}
    (QA/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (QA/'methodology_text.txt').write_text(text,encoding='utf-8')
    with zipfile.ZipFile(HERE/'methodology_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for item in ['report.tex','sections/03_methodology.tex','references_methodology.tex',
                     'generate_methodology.py','build_and_verify.py','build.ps1','README.md',
                     'review_notes.md','figures/methodology_pipeline.pdf',
                     'figures/model_architectures.pdf','qa/methodology_evidence.json']:
            z.write(HERE/item,item)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
