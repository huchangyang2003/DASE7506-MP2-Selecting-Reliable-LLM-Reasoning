"""Soft evidence adapted from Tianyi-325/DASE7506-MP2-main,
8737e0ab5257669e03b14ae6737f0e853c22ab2e, code/math_evidence.py.
V7 keeps the official answer parser and fits no parameters in this module.
"""
import ast,math,re
from fractions import Fraction
import numpy as np
from v7_common import answer_information,parse_answer

NUM=re.compile(r'(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:/\d+(?:\.\d*)?)?(?![\w.])')
WORDS={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
 'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,
 'eighteen':18,'nineteen':19,'twenty':20,'half':.5,'twice':2,'double':2,'dozen':12}
UNIT_GROUPS={
 'time':{'second':1,'seconds':1,'minute':60,'minutes':60,'hour':3600,'hours':3600,'day':86400,'days':86400,'week':604800,'weeks':604800},
 'length':{'inch':.0254,'inches':.0254,'foot':.3048,'feet':.3048,'meter':1,'meters':1,'metre':1,'metres':1,'kilometer':1000,'kilometers':1000,'mile':1609.344,'miles':1609.344},
 'money':{'dollar':1,'dollars':1,'cent':.01,'cents':.01,'euro':1,'euros':1},
 'person':{w:1 for w in ('person','people','student','students','child','children','passenger','passengers','customer','customers','worker','workers')},
 'book':{'book':1,'books':1},'slice':{'slice':1,'slices':1,'piece':1,'pieces':1},
 'animal':{'frog':1,'frogs':1,'dove':1,'doves':1,'bird':1,'birds':1,'cat':1,'cats':1,'dog':1,'dogs':1},
 'area':{'square feet':.09290304,'square foot':.09290304,'square meter':1,'square meters':1},
 'percent':{'percent':1,'%':1}}
UNITS={u:(g,f) for g,d in UNIT_GROUPS.items() for u,f in d.items()}
UNIT_RE=re.compile(r'\bsquare (?:feet|foot|meters?)\b|\b(?:'+ '|'.join(re.escape(u) for u in sorted(UNITS,key=len,reverse=True) if ' ' not in u and u!='%')+r')\b|%',re.I)
STOP_NOUNS={'times','days','weeks','years','hours','minutes','seconds','percent','more','less','of','and','in','to'}
SEM_NAMES=['q_number_operation_coverage','q_number_missing_log','salient_factor_known','salient_factor_missing',
 'salient_factor_used_fraction','target_unit_known','terminal_unit_known','target_unit_match','target_unit_mismatch',
 'target_total_or_now','target_remaining','target_profit','target_each_or_split','target_difference',
 'final_matches_nonlast_result','final_matches_last_result','orphan_result_log','orphan_result_fraction',
 'orphan_result_above_final','q_new_stock_with_final_only_new','q_tax_with_unused_result',
 'q_profit_final_revenue','q_total_final_component','terminal_result_unit_mismatch',
 'candidate_new_number_operand_fraction','named_quantity_reversal','relation_support','relation_conflict']
VERIFY_NAMES=['numeric_checked_log','numeric_wrong_log','numeric_wrong_fraction','max_relative_error_log',
 'unknown_equations_log','no_numeric_checks','symbolic_identity_log','symbolic_nonidentity_homogeneous_log',
 'unit_checked_log','unit_wrong_log','last_box_known','box_official_mismatch','terminal_claim_known',
 'terminal_claim_official_mismatch','marker_looks_like_heading','parse_marker_to_end_log',
 'final_matches_evaluated_value','repair_hypothesis_known','repair_changes_final','division_or_invalid_log',
 'truncated','numeric_checks_per_equal']
REL_NAMES=['arith_quality_rank','condition_coverage_rank','unit_quality_rank','final_quality_rank',
 'pareto_dominates_fraction','pareto_dominated_fraction','same_group_evidence_advantage',
 'minority_with_more_evidence','support_times_arith_wrong','support_times_unit_mismatch',
 'salient_larger_ratio','salient_smaller_ratio','total_larger_ratio','split_smaller_ratio',
 'other_claims_my_official_answer','other_repair_supports_my_answer','own_repair_supports_rival',
 'verified_minority_against_bad_majority','relative_orphan_quality','group_min_wrong_fraction']
LIMIT=1e15

def bounded(v):
    v=Fraction(v)
    if abs(v)>LIMIT or v.denominator>10**15:raise ValueError('out of numeric bounds')
    return v

def number_values(text,words=False):
    out=set()
    for m in NUM.finditer(text):
        try:out.add(bounded(m.group().replace(',','')))
        except (ValueError,ZeroDivisionError,OverflowError):pass
    if words:
        out.update(Fraction(str(v)) for w,v in WORDS.items() if re.search(r'\b'+w+r'\b',text,re.I))
    return out

def close(a,b):return abs(float(a-b))<=1e-3*max(1.,abs(float(a)),abs(float(b)))

def normalize(text):
    s=str(text)[:24000].replace('−','-').replace('×','*').replace('÷','/')
    for old,new in [(r'\times','*'),(r'\cdot','*'),(r'\div','/'),(r'\left',''),(r'\right',''),
                    (r'\,',' '),(r'\;',' '),(r'\!',' '),(r'\quad',' '),(r'\%','%')]:s=s.replace(old,new)
    s=re.sub(r'(?<=\d),(?=\d)','',s)
    s=re.sub(r'(?<![\w.)])\*\*([^*\n]{1,300})\*\*(?![\w.(])',r'\1',s)
    # Balanced TeX fractions; bounded passes, never execute text.
    for _ in range(8):
        m=re.search(r'\\(?:dfrac|tfrac|frac)\s*\{',s)
        if not m:break
        start=m.end()-1
        def brace(pos):
            if pos>=len(s) or s[pos]!='{':raise ValueError()
            depth=0
            for k in range(pos,min(len(s),pos+600)):
                depth+=(s[k]=='{')-(s[k]=='}')
                if depth==0:return s[pos+1:k],k+1
            raise ValueError()
        try:
            a,end=brace(start)
            while end<len(s) and s[end].isspace():end+=1
            b,end2=brace(end);s=s[:m.start()]+'(('+a+')/('+b+'))'+s[end2:]
        except ValueError:break
    s=re.sub(r'\\(?:text|mathrm|mathbf)\{([^{}]*)\}',r' \1 ',s)
    s=re.sub(r'\\boxed\{([^{}]*)\}',r' \1 ',s)
    s=re.sub(r'(?<![\w.])(\d+(?:\.\d+)?)\s*%',r'(\1/100)',s)
    s=s.replace('$','').replace('€','').replace('\\(','').replace('\\)','').replace('\\[','').replace('\\]','')
    return s

def expression(segment,tail=False):
    # Pull a numeric/single-letter expression; words/units separate expressions.
    segment=segment.strip(' \t:;,$')
    segment=UNIT_RE.sub(' ',segment)
    tokens=re.findall(r'[A-Za-z]+|(?:\d+(?:\.\d*)?|\.\d+)|[()+*/^.-]|\s+|.',segment)
    seq=[];source=reversed(tokens) if tail else tokens
    for tok in source:
        if tok.isspace():seq.append(tok);continue
        if re.fullmatch(r'[A-Za-z]',tok) or re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)|[()+*/^.-]',tok):
            seq.append(tok)
        elif not seq and tok in (':',';'):continue
        else:break
    if tail:seq.reverse()
    e=''.join(seq).strip().rstrip('.')
    e=re.sub(r'(?<=\d)\s*([a-zA-Z])\b',r'*\1',e)
    e=re.sub(r'(?<=\))\s*(?=[(\da-zA-Z])','*',e).replace('^','**')
    if not e or len(e)>180:raise ValueError('no bounded expression')
    return e

def polynomial(expr,substitutions=None):
    tree=ast.parse(expr,mode='eval')
    if len(list(ast.walk(tree)))>64:raise ValueError('expression too complex')
    operands=set();vars_used=set()
    def add(a,b,sign=1):
        d=a.copy()
        for k,v in b.items():d[k]=bounded(d.get(k,0)+sign*v)
        return {k:v for k,v in d.items() if v}
    def mul(a,b):
        d={}
        for ma,va in a.items():
            for mb,vb in b.items():
                key=tuple(sorted(ma+mb))
                if len(key)>2:raise ValueError('polynomial degree')
                d[key]=bounded(d.get(key,0)+va*vb)
        return {k:v for k,v in d.items() if v}
    def visit(node,depth=0):
        if depth>14:raise ValueError('too deep')
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            v=bounded(str(node.value));operands.add(v);res={():v} if v else {}
        elif isinstance(node,ast.Name) and len(node.id)==1 and node.id.isalpha():
            vars_used.add(node.id.lower());res={(node.id.lower(),):Fraction(1)}
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            res={k:v*(1 if isinstance(node.op,ast.UAdd) else -1) for k,v in visit(node.operand,depth+1).items()}
        elif isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            a=visit(node.left,depth+1);b=visit(node.right,depth+1)
            if isinstance(node.op,ast.Add):res=add(a,b)
            elif isinstance(node.op,ast.Sub):res=add(a,b,-1)
            elif isinstance(node.op,ast.Mult):res=mul(a,b)
            elif isinstance(node.op,ast.Div):
                if any(k for k in b):raise ValueError('variable denominator')
                den=b.get((),0)
                if not den:raise ZeroDivisionError()
                res={k:bounded(v/den) for k,v in a.items()}
            else:
                if any(k for k in b):raise ValueError('variable exponent')
                exponent=b.get((),0)
                if exponent.denominator!=1 or not 0<=exponent<=2:raise ValueError('power bound')
                res={():Fraction(1)}
                for _ in range(int(exponent)):res=mul(res,a)
        else:raise ValueError('unsafe or unsupported AST')
        if substitutions is not None and not any(k for k in res):
            v=res.get((),Fraction(0))
            if v in substitutions:return {():substitutions[v]}
        return res
    result=visit(tree.body)
    return result,operands,vars_used

def unit_near(s,tail=False):
    found=list(UNIT_RE.finditer(s))
    if not found:return None
    m=found[-1] if tail else found[0]
    # Unit must be close to the adjoining numeric/math expression, not arbitrary narrative.
    if (len(s)-m.end() if tail else m.start())>45:return None
    return UNITS.get(m.group().lower())

def equations(text):
    normalized=normalize(text);steps=[];unknown=bad=unit_checked=unit_wrong=identities=homogeneous=0;operands=set()
    equal=normalized.count('=');invalid=0
    for line in normalized.splitlines():
        if '=' not in line:continue
        pieces=line.split('=')
        for left,right in zip(pieces,pieces[1:]):
            if len(steps)+unknown>=64:break
            try:
                le=expression(left,True);rexp=expression(right,False)
                lp,lo,lv=polynomial(le);rp,ro,rv=polynomial(rexp)
                operands.update(lo);operands.update(ro)
                if lv or rv:
                    diff=lp.copy()
                    for k,v in rp.items():diff[k]=diff.get(k,0)-v
                    diff={k:v for k,v in diff.items() if v}
                    if not diff:identities+=1
                    elif () not in diff and any(k for k in diff):homogeneous+=1
                    else:unknown+=1
                    continue
                l=lp.get((),Fraction(0));r=rp.get((),Fraction(0))
                lu=unit_near(left,True);ru=unit_near(right,False)
                # Mixed scales within an expression need dimensional algebra; abstain.
                us=[UNITS[m.group().lower()] for m in UNIT_RE.finditer(left)]
                vs=[UNITS[m.group().lower()] for m in UNIT_RE.finditer(right)]
                currencies=[m.group().lower() for m in UNIT_RE.finditer(left+' '+right)]
                if len(set(us))>1 or len(set(vs))>1 or (any('euro' in u for u in currencies) and any(u in ('dollar','dollars','cent','cents') for u in currencies)):
                    unknown+=1;continue
                if lu and ru:
                    # Only convert explicit scalar quantities. Compound expressions
                    # require dimensional algebra; e.g. 6 hours * 60 = 360 minutes
                    # must not have the conversion applied a second time.
                    literal=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
                    if not (re.fullmatch(literal,le.strip()) and re.fullmatch(literal,rexp.strip())):
                        unknown+=1;continue
                    lm=list(UNIT_RE.finditer(left));rm=list(UNIT_RE.finditer(right))
                    # The numeric value must directly adjoin its unit. Words
                    # like 'second job' and 'per hour' are not conversion sides.
                    lbefore=left[:lm[-1].start()] if lm else ''
                    lafter=left[lm[-1].end():] if lm else ''
                    rbefore=right[:rm[0].start()] if rm else ''
                    if (not re.search(literal+r'\s*$',lbefore)
                        or lafter.strip(' .,:;')
                        or not re.fullmatch(r'\s*'+literal+r'\s*',rbefore)):
                        unknown+=1;continue
                    if lu[0]!=ru[0]:unknown+=1;continue
                    unit_checked+=1
                    err=not close(l*Fraction(str(lu[1])),r*Fraction(str(ru[1])))
                    unit_wrong+=int(err);steps.append({'left':le,'computed':l,'claimed':r,'wrong':False,'unit':True});continue
                # Do not treat a detached narrative number, such as '(day 2)',
                # as a complete computed expression on the left of an equality.
                if not any(isinstance(node,ast.BinOp) for node in ast.walk(ast.parse(le,mode='eval'))):
                    unknown+=1;continue
                err=abs(float(l-r))/max(1.,abs(float(l)),abs(float(r)))
                steps.append({'left':le,'computed':l,'claimed':r,'wrong':err>1e-3,'error':err,'unit':False})
                bad+=int(err>1e-3)
            except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):
                unknown+=1;invalid+=1
    # A bounded single-error repair hypothesis, used only as soft cross-candidate evidence.
    first=next((k for k,s in enumerate(steps) if s['wrong']),None);repair=None
    if first is not None:
        remap={steps[first]['claimed']:steps[first]['computed']}
        for step in steps[first+1:]:
            if step['unit']:continue
            try:
                p,_,v=polynomial(step['left'],remap)
                if not v:remap[step['claimed']]=bounded(p.get((),0))
                if len(remap)>12:break
            except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):pass
        repair=remap
    return {'steps':steps,'operands':operands,'bad':bad,'unknown':unknown,'equal':equal,'invalid':invalid,
      'unit_checked':unit_checked,'unit_wrong':unit_wrong,'identities':identities,'homogeneous':homogeneous,'repair':repair}

def unit_target(question):
    # A decimal point inside a number is not a sentence boundary.
    q=question.lower();tail=re.split(r'[?!]|(?<!\d)\.|\.(?!\d)',q)
    fragment=next((s for s in reversed(tail) if s.strip()),q)
    if re.search(r'\bhow much\b',fragment) and re.search(r'pay|cost|money|spend|earn|price|profit|change|dollars?',fragment):return 'money'
    if re.search(r'percentage|percent',fragment):return 'percent'
    matches=list(UNIT_RE.finditer(fragment))
    return UNITS[matches[-1].group().lower()][0] if matches else None

def salient_factors(q):
    result=set()
    pattern=r'\b(\d+|'+ '|'.join(WORDS)+r')\s+([a-z]+)'
    for m in re.finditer(pattern,q.lower()):
        token,noun=m.groups()
        try:v=Fraction(str(WORDS[token])) if token in WORDS else Fraction(token)
        except ValueError:continue
        if 1<v<=30 and noun not in STOP_NOUNS:result.add(v)
    for ordinal,n in [('second',2),('third',3),('fourth',4)]:
        if re.search(r'\b'+ordinal+r'\s+(?:coat|layer|round|time|trip)',q,re.I):result.add(Fraction(n))
    return result

def role_relations(text,person):
    # Restricted explicit subject-is-twice/half-object relations; pronoun ambiguity -> no rule.
    out=[]
    pattern=r'\b([A-Za-z]+|(?:her|his|the)\s+(?:sister|brother|father|mother))\s+(?:is|has)\s+(twice|half|double|three times)\s+(?:as\s+)?(?:many|much|the|age|number|of|as|years|old|\s)*([A-Za-z]+|(?:her|his)\s+(?:age|sister|brother|father|mother))'
    for m in re.finditer(pattern,text,re.I):
        subject,ratio,obj=[x.lower().strip() for x in m.groups()]
        subject=re.sub(r'^(?:her|his|the)\s+','',subject)
        if obj in ('her','his') or obj in ('her age','his age'):obj=person
        obj=re.sub(r'^(?:her|his|the)\s+','',obj)
        if subject in ('it','there','she','he') or obj in ('as','the','that','her','his','many','much','old'):continue
        out.append((subject,obj,Fraction(str({'twice':2,'double':2,'half':.5,'three times':3}[ratio]))))
    return out

def candidate_evidence(q,text,answer,context):
    eq=equations(text);steps=[s for s in eq['steps'] if not s['unit']];ops=eq['operands']
    qnums=context['numbers'];factors=context['factors'];target=context['target'];flags=context['flags']
    a=Fraction(answer) if answer is not None else None
    result_values=[s['claimed'] for s in steps]
    consumed=set()
    for k,s in enumerate(steps):
        try:
            _,op,_=polynomial(s['left'])
            consumed.update(v for v in result_values[:k] if v in op)
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):pass
    orphan=[v for v in set(result_values) if v not in consumed and v!=a and abs(v)>1]
    final_last=a is not None and bool(result_values) and close(a,result_values[-1])
    final_prior=a is not None and any(close(a,v) for v in result_values[:-1]) and not final_last
    boxes=re.findall(r'\\boxed\{([^{}]+)\}',text)
    box=parse_answer(r'\boxed{'+boxes[-1]+'}') if boxes else None
    claim=box
    if claim is None:
        m=re.search(r'(?:final answer|answer)\s*[:：]?\s*[*$\s]*([-+]?\d[\d,.]*(?:/\d+)?)',text[-500:],re.I)
        if m:claim=parse_answer('#### '+m.group(1))
    # Unit evidence refers to the terminal claim, not an earlier unit in the
    # last 450 characters. Missing explicit terminal units remain unknown.
    if '####' in text:
        final_chunk=text.rsplit('####',1)[-1][:450]
    else:
        terminal_lines=[line.strip() for line in text.splitlines()
                        if line.strip() and line.strip() not in ('$$','$',r'\[',r'\]')]
        final_chunk=terminal_lines[-1][-450:] if terminal_lines else ''
    um=list(UNIT_RE.finditer(final_chunk))
    currency=final_chunk.count('$')==1 and re.search(r'\$\s*\d',final_chunk) is not None
    unit=UNITS[um[-1].group().lower()][0] if um else ('money' if currency else None)
    used={abs(v) for v in ops};qn={abs(v) for v in qnums};missing=qn-used
    sfmissing=factors-used
    person=context['person'];rels=role_relations(text,person);support=conflict=0
    for s,o,r in context['relations']:
        support+=sum(cs==s and co==o and cr==r for cs,co,cr in rels)
        conflict+=sum((cs==o and co==s and cr==r and r!=1) or (cs==s and co==o and cr!=r) for cs,co,cr in rels)
    terminal_label=' '.join(text.lower().splitlines()[-6:])
    final_revenue=bool(re.search(r'revenue|sales|earned from selling|total earnings',terminal_label))
    sem=[len(qn&used)/max(1,len(qn)),math.log1p(len(missing)),float(bool(factors)),float(bool(sfmissing)),
      len(factors&used)/max(1,len(factors)),float(target is not None),float(unit is not None),
      float(target is not None and unit==target),float(target is not None and unit is not None and unit!=target),
      *flags, float(final_prior),float(final_last),math.log1p(len(orphan)),len(orphan)/max(1,len(set(result_values))),
      float(a is not None and any(v>a for v in orphan)),
      float(flags[0] and re.search(r'hatch|born|new |grow up|survive',q,re.I) is not None and final_last),
      float(re.search(r'tax',q,re.I) is not None and bool(orphan)),
      float(flags[2] and final_revenue),float(flags[0] and final_prior),
      float(target is not None and unit is not None and target!=unit and final_last),
      len(used-qn)/max(1,len(used)),float(conflict>0),float(support>0),float(conflict)]
    repair=eq['repair'].get(a) if eq['repair'] is not None and a is not None else None
    if repair is not None and close(repair,a):repair=None
    marker=text.rsplit('####',1)[-1] if '####' in text else ''
    heading=bool(marker and re.match(r'\s*[* ]*(?:job|step|day|part|case|example)\s+\d',marker,re.I))
    ver=[math.log1p(len(steps)),math.log1p(eq['bad']),eq['bad']/max(1,len(steps)),
      math.log1p(max((s.get('error',0) for s in steps),default=0)),math.log1p(eq['unknown']),
      float(not steps),math.log1p(eq['identities']),math.log1p(eq['homogeneous']),
      math.log1p(eq['unit_checked']),math.log1p(eq['unit_wrong']),float(box is not None),
      float(box is not None and box!=answer),float(claim is not None),float(claim is not None and claim!=answer),
      float(heading),math.log1p(len(marker)) if marker else 0.,
      float(a is not None and any(close(a,s['computed']) for s in steps)),float(eq['repair'] is not None),
      float(repair is not None),math.log1p(eq['invalid']),float(len(text)>24000),len(steps)/max(1,eq['equal'])]
    assert len(sem)==len(SEM_NAMES) and len(ver)==len(VERIFY_NAMES),(len(sem),len(ver))
    return sem,ver,{'claim':Fraction(claim) if claim is not None else None,'repair':repair,'parsed':a,
      'coverage':sem[0],'wrong':ver[2]+float(eq['unit_wrong']>0),'unitquality':sem[7]-sem[8],
      'finalquality':float(final_last)-ver[11]-ver[13],'orphan':sem[17],'steps':len(steps)}

def extract_evidence(bank):
    n=len(bank);sem=np.zeros((n,4,len(SEM_NAMES)),np.float64);ver=np.zeros((n,4,len(VERIFY_NAMES)),np.float64)
    rel=np.zeros((n,4,len(REL_NAMES)),np.float64);answers,counts,votes=answer_information(bank);diagnostics=[]
    for i,row in enumerate(bank):
        q=row['question'];qlo=q.lower()
        flags=[float(bool(re.search(p,qlo))) for p in
          (r'total|in all|altogether|now|all the|all of',r'remain|left|change|rest',
           r'profit|gain|net ',r'each.*contribute|per person|split|each pay',
           r'difference|how much more|how many more')]
        names=re.findall(r'\b[A-Z][a-z]{2,}\b',q);person=names[0].lower() if names else ''
        context={'numbers':number_values(q,True),'factors':salient_factors(q),'target':unit_target(q),
                 'flags':flags,'person':person,'relations':role_relations(q,person)}
        meta=[]
        for j,c in enumerate(row['candidates']):
            sf,vf,md=candidate_evidence(q,c['text'],answers[i][j],context);sem[i,j]=sf;ver[i,j]=vf;meta.append(md)
        for j,m in enumerate(meta):
            rivals=[k for k in range(4) if answers[i][k] is not None and answers[i][k]!=answers[i][j]]
            if not rivals:rivals=[k for k in range(4) if k!=j]
            rank=lambda key,better=1:sum(better*(m[key]-meta[k][key])>1e-8 for k in rivals)/max(1,len(rivals))
            dominates=dominated=0
            for k in rivals:
                x=[m['coverage']-meta[k]['coverage'],meta[k]['wrong']-m['wrong'],m['unitquality']-meta[k]['unitquality']]
                dominates+=int(min(x)>=-1e-8 and max(x)>1e-8);dominated+=int(max(x)<=1e-8 and min(x)<-1e-8)
            larger=smaller=0
            if m['parsed'] is not None:
                for k in rivals:
                    other=meta[k]['parsed']
                    if other is None or not other:continue
                    ratio=abs(m['parsed']/other)
                    larger+=int(any(close(ratio,f) for f in context['factors']))
                    smaller+=int(ratio!=0 and any(close(1/ratio,f) for f in context['factors']))
            others_claim=sum(m['parsed'] is not None and meta[k]['claim']==m['parsed'] for k in rivals)
            repair_support=sum(m['parsed'] is not None and meta[k]['repair'] is not None and close(meta[k]['repair'],m['parsed']) for k in rivals)
            own_support=sum(m['repair'] is not None and meta[k]['parsed'] is not None and close(m['repair'],meta[k]['parsed']) for k in rivals)
            group=[k for k in range(4) if answers[i][k]==answers[i][j] and answers[i][j] is not None] or [j]
            adv=m['coverage']-np.mean([meta[k]['coverage'] for k in rivals])
            minority=votes[i,j]<votes[i].max()
            rel[i,j]=[rank('wrong',-1),rank('coverage'),rank('unitquality'),rank('finalquality'),
              dominates/max(1,len(rivals)),dominated/max(1,len(rivals)),adv,float(minority)*adv,
              votes[i,j]*m['wrong'],votes[i,j]*float(sem[i,j,8]>0),larger/max(1,len(rivals)),smaller/max(1,len(rivals)),
              flags[0]*larger/max(1,len(rivals)),flags[3]*smaller/max(1,len(rivals)),others_claim/3,repair_support/3,
              own_support/3,float(minority and m['wrong']==0 and any(meta[k]['wrong']>0 and votes[i,k]>votes[i,j] for k in rivals)),
              rank('orphan',-1),min(meta[k]['wrong'] for k in group)]
        diagnostics.append({'numeric_checked':sum(m['steps'] for m in meta),
          'target_unit_known':context['target'] is not None,'salient_factors':len(context['factors'])})
    assert np.isfinite(sem).all() and np.isfinite(ver).all() and np.isfinite(rel).all()
    return {'semantic':sem,'verify':ver,'relative':rel},diagnostics
