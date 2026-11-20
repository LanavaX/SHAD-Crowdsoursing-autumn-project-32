"""Normalize bbox output, construct review records, adjudicate without losing lineage."""
from __future__ import annotations
import math

def normalized_boxes(shapes: list[dict]) -> list[dict]:
    result=[]
    for s in shapes:
        if s.get('shape',s.get('type','rectangle'))!='rectangle':raise ValueError('Rectangle required')
        if all(k in s for k in ('left','top','width','height')):
            l,t,w,h=(s[k] for k in ('left','top','width','height'))
        elif isinstance(s.get('data'),dict) and 'p1' in s['data']:
            a,b=s['data']['p1'],s['data']['p2'];l,t=min(a['x'],b['x']),min(a['y'],b['y']);w,h=abs(a['x']-b['x']),abs(a['y']-b['y'])
        else:raise ValueError('Unknown annotation format; inspect actual preview output first')
        if any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in (l,t,w,h)):raise ValueError('Invalid number')
        if not(0<=l<1 and 0<=t<1 and w>0 and h>0 and l+w<=1.000001 and t+h<=1.000001):raise ValueError('Box outside frame')
        result.append({'left':l,'top':t,'width':min(w,1-l),'height':min(h,1-t)})
    return result

def bbox_answer(output: dict) -> tuple[str,list[dict]]:
    status=output['status'];boxes=normalized_boxes(output.get('boxes') or [])
    if status=='PRESENT' and not boxes:raise ValueError('PRESENT requires boxes')
    if status in ('ABSENT','UNCERTAIN','IMAGE_ERROR') and boxes:raise ValueError('Stale boxes conflict with status')
    if status not in ('PRESENT','ABSENT','UNCERTAIN','IMAGE_ERROR'):raise ValueError('Unknown status')
    return status,boxes

def review_decision(author_id: str,votes: list[dict]) -> str:
    if any(v['worker_id']==author_id for v in votes):raise ValueError('Self-review')
    if len({v['worker_id'] for v in votes})!=len(votes):raise ValueError('Repeated reviewer')
    if len(votes)<2:return 'pending'
    if all(v['verdict']=='OK' for v in votes):return 'accepted_pending_audit'
    return 'manual_adjudication'

if __name__=='__main__':
    assert normalized_boxes([{'type':'rectangle','data':{'p1':{'x':.7,'y':.8},'p2':{'x':.2,'y':.3}}}])[0]['left']==.2
    for bad in [{'status':'PRESENT','boxes':[]},{'status':'ABSENT','boxes':[{'left':0,'top':0,'width':.5,'height':.5}]}]:
        try:bbox_answer(bad)
        except ValueError:pass
        else:raise AssertionError('Conflicting answer accepted')
    assert bbox_answer({'status':'ABSENT','boxes':[]})==('ABSENT',[])
    assert review_decision('a',[{'worker_id':'b','verdict':'OK'},{'worker_id':'c','verdict':'BAD'}])=='manual_adjudication'
    try:review_decision('a',[{'worker_id':'a','verdict':'OK'}])
    except ValueError:pass
    else:raise AssertionError('Self-review accepted')
    print('Coordinate, status, independent-review checks passed')
