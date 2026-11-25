"""Generate a combined gaze path, direction, density and dwell visualization."""
import json
from pathlib import Path

import cv2
import numpy as np


def _density_dwells(density, data, dt, scale, radius=100, minimum_seconds=.4):
    """Find dwell peaks even when occasional gaze outliers interrupt a fixation."""
    work=density.copy()
    peak=float(work.max())
    groups=[]
    if peak<=0:
        return groups
    exclusion=max(1,int(140/scale))
    for _ in range(12):
        y,x=np.unravel_index(np.argmax(work),work.shape)
        if work[y,x] < .12*peak:
            break
        center=np.array([(x+.5)*scale,(y+.5)*scale])
        distance=np.linalg.norm(data[:,1:3]-center,axis=1)
        duration=float(dt[distance<=radius].sum())
        count=int((distance<=radius).sum())
        if duration>=minimum_seconds:
            weighted=data[distance<=radius,1:3]
            center=np.median(weighted,axis=0)
            groups.append(dict(x=float(center[0]),y=float(center[1]),
                               duration_s=duration,samples=count))
        cv2.circle(work,(x,y),exclusion,0,-1)
    return groups


def build_gaze_visualization(records, width, height):
    if len(records)<2:
        return None, {'samples':len(records),'duration_s':0.,'dwells':[]}
    data=np.asarray(records,dtype=float)
    data[:,1]=np.clip(data[:,1],0,width-1)
    data[:,2]=np.clip(data[:,2],0,height-1)
    unclipped=((data[:,1]>2)&(data[:,1]<width-3)&(data[:,2]>2)&(data[:,2]<height-3))
    clipped_count=int((~unclipped).sum())
    data=data[unclipped]
    if len(data)<2:
        return None,{'samples':len(records),'usable_samples':len(data),
                     'clipped_boundary_samples':clipped_count,'duration_s':0.,'dwells':[]}
    scale=4
    sw,sh=max(1,width//scale),max(1,height//scale)
    density=np.zeros((sh,sw),np.float32)
    dt=np.diff(data[:,0],append=data[-1,0]+np.median(np.diff(data[:,0])))
    dt=np.clip(dt,.01,.5)
    for (_,x,y),weight in zip(data,dt):
        density[min(sh-1,int(y/scale)),min(sw-1,int(x/scale))]+=float(weight)
    density=cv2.GaussianBlur(density,(0,0),sigmaX=18,sigmaY=18)
    peak=float(np.quantile(density[density>0],.995)) if np.any(density>0) else 0.
    normalized=(np.zeros_like(density,dtype=np.uint8) if peak<=0 else
                np.uint8(np.clip(np.sqrt(density/peak)*255,0,255)))
    normalized=cv2.resize(normalized,(width,height),interpolation=cv2.INTER_LINEAR)
    heat=cv2.applyColorMap(normalized,cv2.COLORMAP_JET)
    strength=normalized.astype(np.float32)/255
    alpha=np.where(strength>0,.18+.77*strength,0)[...,None]
    image=np.uint8(255*(1-alpha)+heat*alpha)
    points=np.rint(data[:,1:3]).astype(int)
    smooth=np.array([np.median(points[max(0,i-2):i+3],axis=0) for i in range(len(points))])
    step=max(1,len(smooth)//180)
    shown=np.rint(smooth[::step]).astype(int)
    path_layer=image.copy()
    jump_limit=.15*np.hypot(width,height)
    for i in range(1,len(shown)):
        if np.linalg.norm(shown[i]-shown[i-1])>jump_limit:
            continue
        ratio=i/max(1,len(shown)-1)
        color=(int(235*(1-ratio)),int(80+150*ratio),int(40+180*ratio))
        cv2.line(path_layer,tuple(shown[i-1]),tuple(shown[i]),color,1,cv2.LINE_AA)
    arrow_step=max(2,len(shown)//14)
    for i in range(arrow_step,len(shown),arrow_step):
        if np.linalg.norm(shown[i]-shown[i-arrow_step])<=jump_limit:
            cv2.arrowedLine(path_layer,tuple(shown[i-arrow_step]),tuple(shown[i]),
                            (35,35,35),1,cv2.LINE_AA,tipLength=.18)
    image=cv2.addWeighted(image,.72,path_layer,.28,0)
    dwells=_density_dwells(density,data,dt,scale)
    for index,dwell in enumerate(sorted(dwells,key=lambda d:d['duration_s'],reverse=True)[:12],1):
        center=(int(dwell['x']),int(dwell['y']))
        radius=int(np.clip(16+dwell['duration_s']*7,18,55))
        cv2.circle(image,center,radius,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,f'{index}: {dwell["duration_s"]:.1f}s',(center[0]+radius+4,center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),2,cv2.LINE_AA)
    cv2.rectangle(image,(12,12),(510,82),(255,255,255),-1)
    cv2.putText(image,'GAZE PATH: blue -> green -> magenta | arrows: direction',(22,38),
                cv2.FONT_HERSHEY_SIMPLEX,.53,(30,30,30),1,cv2.LINE_AA)
    cv2.putText(image,'HEAT: dwell density | white circles: dwell duration',(22,66),
                cv2.FONT_HERSHEY_SIMPLEX,.53,(30,30,30),1,cv2.LINE_AA)
    summary={'samples':len(records),'usable_samples':len(data),
             'clipped_boundary_samples':clipped_count,
             'duration_s':float(data[-1,0]-data[0,0]),
             'dwells':dwells,'coordinate_system':'displayed filtered screen pixels',
             'excluded':'calibration and validation stages'}
    return image,summary


def save_gaze_visualization(records, width, height, folder):
    image,summary=build_gaze_visualization(records,width,height)
    if image is None:
        return None,summary
    folder=Path(folder)
    output=folder/'gaze_path_heatmap.png'
    ok,encoded=cv2.imencode('.png',image)
    if not ok:
        raise RuntimeError('Failed to encode gaze heatmap')
    encoded.tofile(str(output))
    (folder/'gaze_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return output,summary
