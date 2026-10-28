"""Validate HTML structure and execute its own controls in a headless renderer."""
from pathlib import Path
from html.parser import HTMLParser
import json
import re
import subprocess
import sys
import tempfile
import shutil

HERE = Path(__file__).resolve().parent
QA = HERE / 'qa'
QA.mkdir(exist_ok=True)
document = HERE / 'lstm_transformer_study_guide.html'
text = document.read_text(encoding='utf-8')


class Structure(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.links = []
        self.images = []
        self.visible = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            self.ids.append(a['id'])
        if tag == 'a':
            self.links.append(a.get('href',''))
        if tag == 'img':
            self.images.append(a)
        if tag in ('style', 'script'):
            self.hidden_depth += 1

    def handle_endtag(self, tag):
        if tag in ('style', 'script'):
            self.hidden_depth -= 1

    def handle_data(self, data):
        if not self.hidden_depth:
            self.visible.append(data)


parser = Structure()
parser.feed(text)
assert len(parser.ids) == len(set(parser.ids)), 'duplicate HTML ids'
for href in parser.links:
    if href.startswith('#'):
        assert href[1:] in parser.ids, href
    elif href and not href.startswith(('https://','http://')):
        assert (HERE / href).exists(), href
assert all(img.get('alt') and img.get('src','').startswith('data:image/png;base64,') for img in parser.images)
assert text.count('class="chapter"') == 20
assert text.count('class="qa"') == 51
assert '__QA__' not in text
assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=["\']https?://',text)

# Test code is appended to a QA-only copy; the delivered HTML has no test hooks.
fixture_script = r'''
const failures=[];const checks=[];
function check(name,condition){checks.push(name);if(!condition)failures.push(name)}
window.addEventListener('error',e=>failures.push('runtime: '+e.message));
window.addEventListener('load',()=>{
 try{
 check('20 chapters',document.querySelectorAll('.chapter').length===20);
 check('51 oral questions',document.querySelectorAll('.qa').length===51);
 check('samples initialized',$('sample-select').options.length===7);
 check('gate arithmetic',Math.abs(0.9*0.8+0.2*(-0.5)-0.62)<1e-10);
 check('gate output shows 0.6200',$('gate-result').textContent.includes('c_new = 0.6200'));
 $('gc').value='0.8';$('gf').value='1';$('gi').value='0';$('go').value='0';$('go').dispatchEvent(new Event('input'));
 check('retained memory and closed output',$('gate-result').textContent.includes('c_new = 0.8000; h_new = 0.0000'));
 $('gc').value='0.8';$('gf').value='0.9';$('gi').value='0.2';$('go').value='0.7';gate();
 $('qx').value='0';$('qy').value='0';$('qx').dispatchEvent(new Event('input'));
 check('equal attention scores',Math.abs(softmax([0,0,0])[0]-1/3)<1e-10 && $('attention-result').textContent.includes('output=(1.0000, 1.0000)'));
 $('qx').value='1';$('qy').value='0';attention();
 check('worked attention output',$('attention-result').textContent.includes('output=(1.2033, 0.7967)'));
 $('representation').value='patch';$('representation').dispatchEvent(new Event('input'));$('token-index').value='26';$('token-index').dispatchEvent(new Event('input'));
 check('patch position mapping',$('token-location').textContent==='Patch 26: rows 12–15, columns 20–23');
 const sample=data.samples[0];const expected=(sample.pixels[28*13+22]/255-data.mean)/data.std;
 check('patch normalized feature 6',$('token-values').textContent.includes(' 6: '+fmt(expected)));
 $('representation').value='row';$('representation').dispatchEvent(new Event('input'));$('token-index').value='12';drawImage();
 check('row representation',$('token-shape').textContent.includes('28 features'));
 $('curve-model').value='transformer';$('curve-model').dispatchEvent(new Event('change'));
 check('transformer epoch selection',document.querySelector('.selected-row td').textContent.startsWith('10'));
 $('curve-model').value='lstm';history();
 check('lstm epoch selection',document.querySelector('.selected-row td').textContent.startsWith('9'));
 $('qa-search').value='forget gate';$('qa-search').dispatchEvent(new Event('input'));
 check('question filtering',document.querySelectorAll('.qa:not([hidden])').length>0 && document.querySelectorAll('.qa:not([hidden])').length<51);
 $('qa-search').value='';filterQA();
 $('expand-all').click();check('open answers',Array.from(document.querySelectorAll('details')).every(el=>el.open));
 $('collapse-all').click();check('close answers',Array.from(document.querySelectorAll('details')).every(el=>!el.open));
 const overflows=Array.from(document.querySelectorAll('body *')).filter(el=>el.getBoundingClientRect().right>innerWidth+2 && !el.closest('pre,.equation,.table-wrap') && getComputedStyle(el).position!=='fixed').map(el=>el.tagName+'#'+el.id);
 check('no page horizontal overflow',document.documentElement.scrollWidth<=innerWidth+2);
 const result={status:failures.length?'failed':'passed',checks,failures,viewport:[innerWidth,innerHeight],scrollWidth:document.documentElement.scrollWidth,overflowElements:overflows};
 const pre=document.createElement('pre');pre.id='qa-result';pre.style.display='none';pre.textContent=JSON.stringify(result);document.body.appendChild(pre);
 const anchor=document.getElementById(location.hash.slice(1));
 if(anchor)anchor.scrollIntoView({behavior:'instant',block:'start'});
 }catch(e){const pre=document.createElement('pre');pre.id='qa-result';pre.textContent=JSON.stringify({status:'failed',error:String(e),stack:e.stack});document.body.appendChild(pre)}
});
'''
fixture = QA / 'render_fixture.html'
fixture.write_text(text.replace('</head>', '<style>html{scroll-behavior:auto!important}</style></head>').replace('</body>', '<script>'+fixture_script+'</script></body>'),encoding='utf-8')
chrome = Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')
assert chrome.exists()
report = {'structure':'passed', 'chapters':20, 'oral_questions':51, 'embedded_images':len(parser.images), 'approximate_words':len(re.findall(r"\b[\w'-]+\b",' '.join(parser.visible))), 'renders':[]}
profile = Path(tempfile.mkdtemp(prefix='a1-study-guide-chrome-')).resolve()
for name,width,height,fragment in [('desktop',1440,1100,''),('image-lab',1440,1100,'image-lab'),('attention',1440,1100,'attention'),('mobile',516,1027,'')]:
    # Show a selected chapter at the top in the rendering fixture. This avoids
    # compositor blank captures after very long programmatic scrolls in Chrome.
    # No presentation CSS is changed in the delivered HTML.
    focus_style=''
    if name=='image-lab':
        focus_style='header.hero,section.chapter,footer{display:none}section#data{display:block;padding-top:0;border:0}section#data>*{display:none}section#data>#image-lab{display:block;margin-top:0}'
    elif name=='attention':
        focus_style='header.hero,section.chapter,footer{display:none}section#attention{display:block;padding-top:0;border:0}section#attention>*{display:none}section#attention>.lab{display:block;margin-top:0}'
    fixture.write_text(text.replace('</head>', '<style>html{scroll-behavior:auto!important}'+focus_style+'</style></head>').replace('</body>', '<script>'+fixture_script+'</script></body>'),encoding='utf-8')
    args=[str(chrome),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--hide-scrollbars','--allow-file-access-from-files',f'--user-data-dir={profile}',f'--window-size={width},{height}','--force-device-scale-factor=1','--run-all-compositor-stages-before-draw','--virtual-time-budget=1800',f'--screenshot={QA / (name+".png")}', '--dump-dom',fixture.as_uri()]
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    dom=result.stdout.decode('utf-8',errors='replace')
    match=re.search(r'<pre id="qa-result"[^>]*>(.*?)</pre>',dom,re.S)
    assert match, result.stderr.decode(errors='replace')[-2000:]
    # JSON textContent is HTML-escaped in a serialized DOM.
    from html import unescape
    audit=json.loads(unescape(match.group(1)))
    audit['name']=name
    report['renders'].append(audit)
    assert audit['status']=='passed', audit
    assert (QA / (name+'.png')).exists()
    print(name+': passed',flush=True)
(QA / 'document_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
assert profile.parent == Path(tempfile.gettempdir()).resolve()
assert profile.name.startswith('a1-study-guide-chrome-')
shutil.rmtree(profile)
assert fixture.resolve().parent == QA.resolve()
fixture.unlink()
print(json.dumps({k:v for k,v in report.items() if k!='renders'},indent=2))
