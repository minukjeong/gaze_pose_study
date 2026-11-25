"""Summarize matched 12-target measurements by instructed posture and pass.

No statistical significance claim: a single user and repeated frames are not independent.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def summarize(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        if r.get('settled_1s') != '1' or r.get('valid') != '1':
            continue
        groups[(r.get('posture_label','unlabeled'), r['pass_id'], r['target_id'])].append(r)
    out=[]
    for (posture,pass_id,target), group in sorted(groups.items()):
        rec={'posture':posture,'pass_id':pass_id,'target_id':target,'valid_frames':len(group)}
        metric={'base':'base_distance_image_px', 'selected':'distance_image_px',
                'head':'head_distance_image_px','head_shoulders':'shoulders_distance_image_px'}
        for k,col in metric.items():
            vals=[float(r[col]) for r in group if r.get(col,'') not in ('',None)]
            rec[k+'_px']=round(float(np.mean(vals)),4) if vals else ''
            rec[k+'_coverage']=round(len(vals)/len(group),4)
        # Strict paired comparison: each route gets its own matched-frame baseline.
        for k in ('head','head_shoulders'):
            col=metric[k]
            paired=[r for r in group if r.get(col,'') not in ('',None) and r.get(metric['base'],'') not in ('',None)]
            rec[k+'_paired_n']=len(paired)
            rec[k+'_paired_delta_px']=round(float(np.mean([float(r[metric['base']])-float(r[col]) for r in paired])),4) if paired else ''
        out.append(rec)
    if not out:
        raise SystemExit('No settled, valid target rows. Confirm S was pressed and targets were recorded.')
    dest=Path(path).with_name('ablation_by_target.csv')
    with open(dest,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print('Saved:',dest,'target/pass groups:',len(out))
    for posture in sorted(set(r['posture'] for r in out)):
        a=[r for r in out if r['posture']==posture]
        print('\n',posture,'targets:',len(a),'passes:',len(set(r['pass_id'] for r in a)))
        for label in ('base','selected','head','head_shoulders'):
            values=[float(r[label+'_px']) for r in a if r[label+'_px']!='']
            print(f'  {label:15s} mean of per-target means={np.mean(values):.2f} px  observed targets={len(values)}' if values else f'  {label}: no output')
        for k in ('head','head_shoulders'):
            diffs=[float(r[k+'_paired_delta_px']) for r in a if r[k+'_paired_delta_px']!='']
            print(f'  {k} matched benefit (base-candidate): {np.mean(diffs):+.2f} px across {len(diffs)} targets' if diffs else f'  {k}: no paired rows')
    print('\nCAUTION: Target rows and frames are correlated. These are descriptive results, not independent participant-level statistics.')
    print('Candidates use extrapolate=True for diagnosis, not approved validation. Missing candidate predictions are not scored as success.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('csv_path',type=Path);summarize(ap.parse_args().csv_path)
