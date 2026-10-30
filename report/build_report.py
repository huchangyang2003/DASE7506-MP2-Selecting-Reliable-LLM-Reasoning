"""Build the publication report from recorded results; never train or select models."""
from html import escape
import argparse
import json
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
args=argparse.ArgumentParser()
args.add_argument('--output',type=Path,default=ROOT/'output/pdf/DASE7506_MP2_Final_Report.pdf')
args.add_argument('--markdown',type=Path,default=ROOT/'report/REPORT.md')
opts=args.parse_args()
OUT=opts.output;OUT.parent.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
A_RESULT=load(ROOT/'evaluation/final_test/test-score.json')
CE_RESULT=load(ROOT/'evaluation/ce_control/test-score.json')
assert A_RESULT['correct']==393 and CE_RESULT['correct']==384
PRESERVED_AI="h('AI assistance and reuse')\np('AI assistance was used for code, experiment orchestration, arithmetic parsing, interpretation, case review and this report. I reviewed the relevant code, independently conducted the experiments, verified the reported results, and remain fully responsible for the final implementation and all associated claims. Official parser, data and evaluator are unchanged. Candidate representations were supplied by the course; their Qwen3-0.6B provenance is in the original manifest. No Qwen weights, external training data or new candidate generations were used. GSM8K attribution and MIT license are retained.')\n"
PRESERVED_REFERENCES='h(\'References\')\np(\'[1] <a href="https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md">Course guide and evaluation rules.</a><br/>[2] <a href="https://github.com/xudongwu-0/DASE7506-MP2/blob/main/code/README.md">Course installation and evaluation instructions.</a><br/>[3] Zaheer et al. <a href="https://arxiv.org/abs/1703.06114">Deep Sets.</a> NeurIPS 2017.<br/>[4] <a href="https://github.com/openai/grade-school-math">GSM8K source and retained MIT license.</a>\',True)'
for name,file in [('Arial','arial.ttf'),('ArialBold','arialbd.ttf'),('ArialItalic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='ArialItalic',boldItalic='ArialBold')
INK=colors.HexColor('#202020');GREY=colors.HexColor('#666666');RULE=colors.HexColor('#B9B9B9')
MARGIN=56.7;WIDTH=A4[0]-2*MARGIN;PAGES=10
styles={
 'body':ParagraphStyle('body',fontName='Arial',fontSize=10.5,leading=14.2,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Arial',fontSize=8.5,leading=11.2,textColor=GREY,spaceAfter=7),
 'title':ParagraphStyle('title',fontName='ArialBold',fontSize=18,leading=23,textColor=INK,spaceAfter=12),
 'head':ParagraphStyle('head',fontName='ArialBold',fontSize=15,leading=20,textColor=INK,spaceAfter=12),
 'sub':ParagraphStyle('sub',fontName='ArialBold',fontSize=11,leading=15,textColor=INK,spaceBefore=8,spaceAfter=6),
 'cell':ParagraphStyle('cell',fontName='Arial',fontSize=9.0,leading=12.0,textColor=INK),
 'cellhead':ParagraphStyle('cellhead',fontName='ArialBold',fontSize=9.0,leading=12.0,textColor=INK),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=7.8,leading=10.5,textColor=INK,spaceAfter=8),
}
story=[];markdown=[]
def p(text,small=False):
    story.append(Paragraph(text,styles['small' if small else 'body']))
    md=re.sub(r'<a href="([^"]+)">([^<]+)</a>',r'[\2](\1)',text)
    markdown.append(re.sub('<[^>]+>','',md.replace('<br/>','\n')).replace('&lt;','<').replace('&gt;','>')+'\n')
def h(text):
    story.append(Paragraph(text,styles['sub']));markdown.append('### '+text+'\n')
def page(title,first=False):
    if not first:story.append(PageBreak())
    story.append(Paragraph(title,styles['title' if first else 'head']))
    markdown.append(('# ' if first else '## ')+title+'\n')
def table(headers,rows,ratios=None):
    ratios=ratios or [1]*len(headers)
    widths=[WIDTH*r/sum(ratios) for r in ratios]
    data=[[Paragraph(escape(str(v)),styles['cellhead' if i==0 else 'cell']) for v in row] for i,row in enumerate([headers]+rows)]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EFEFEF')),
      ('LINEABOVE',(0,0),(-1,0),.6,INK),('LINEBELOW',(0,0),(-1,0),.5,RULE),('LINEBELOW',(0,-1),(-1,-1),.5,RULE),
      ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
      ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,9)])
    markdown.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
    markdown.extend('| '+' | '.join(str(v) for v in row)+' |' for row in rows);markdown.append('')
def code(text):
    story.append(Preformatted(text,styles['code']));markdown.extend(['```powershell',text,'```',''])
def results(rows,key):
    return [[r[key],f"{r['accuracy_mean']:.2f} +/- {r['accuracy_sample_std']:.2f}",f"{r['disagreement_mean']:.2f} +/- {r['disagreement_sample_std']:.2f}"] for r in rows]

exec(compile((ROOT/'report/experimental_sections.py').read_text(encoding='utf-8'),str(ROOT/'report/experimental_sections.py'),'exec'),globals())

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Arial',8);canvas.setFillColor(GREY)
    canvas.drawRightString(A4[0]-MARGIN,23,f'{doc.page} / {PAGES}');canvas.restoreState()
doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,topMargin=48,bottomMargin=44,
 title='Selecting Reliable LLM Reasoning: Method and Evidence',author='DASE7506A coursework; AI assistance disclosed')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(OUT)
if len(reader.pages)!=PAGES:raise RuntimeError(f'Expected {PAGES} pages; got {len(reader.pages)}. Repair layout before delivery.')
opts.markdown.parent.mkdir(parents=True,exist_ok=True)
opts.markdown.write_text('\n'.join(markdown)+'\n',encoding='utf-8')
print(json.dumps({'pdf':str(OUT),'pages':len(reader.pages),'test_correct':393,'test_accuracy':84.1541755888651}))
