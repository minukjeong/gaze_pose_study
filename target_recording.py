"""User-started recording with fixed three-second target presentation."""
import atexit
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TARGETS = [(480,238),(1382,842),(1498,302),(422,778),(1190,194),(730,886),
           (346,454),(1574,626),(806,324),(1114,756),(576,626),(1344,454)]
FIELDS = ['model','session_id','timestamp_unix','pass_id','target_id','target_age_s',
          'settled_1s','valid','target_x_px','target_y_px','gaze_x_px','gaze_y_px',
          'dx_px','dy_px','distance_px','target_x_image_px','target_y_image_px',
          'gaze_x_image_px','gaze_y_image_px','distance_image_px',
          'displayed_x_px','displayed_y_px','canvas_width','canvas_height',
          'base_gaze_x_px','base_gaze_y_px','base_distance_px','base_distance_image_px',
          'pose_correction_status','pose_policy','pose_delta_x_px','pose_delta_y_px',
          'posture_label','head_gaze_x_px','head_gaze_y_px','head_distance_image_px',
          'shoulders_gaze_x_px','shoulders_gaze_y_px','shoulders_distance_image_px',
          'head_correction_status','shoulders_correction_status']


def pair(value):
    try:
        return len(value)==2 and all(math.isfinite(float(x)) for x in value)
    except (TypeError,ValueError):
        return False


class TargetRecorder:
    def __init__(self, width, height, model, output_root=None, image=None):
        import cv2
        import numpy as np
        self.width,self.height=int(width),int(height)
        self.model=model
        self.image=Path(image or Path(__file__).with_name('gaze_targets.png')).resolve()
        original=cv2.imdecode(np.fromfile(str(self.image),np.uint8),cv2.IMREAD_COLOR)
        if original is None or original.shape[:2]!=(1080,1920):
            raise ValueError('Expected the supplied 1920x1080 gaze_targets.png')
        # Verify the numbered target centers before any measurement is made.
        for x,y in TARGETS:
            b,g,r=map(int,original[y,x])
            if not (r>180 and g<100 and b<100):
                raise ValueError('Target image does not match the 12 known centers')
        scale=min(self.width/1920,self.height/1080)
        rw,rh=max(1,round(1920*scale)),max(1,round(1080*scale))
        self.sx,self.sy=rw/1920,rh/1080
        self.left,self.top=(self.width-rw)//2,(self.height-rh)//2
        self.background=np.full((self.height,self.width,3),245,np.uint8)
        self.background[self.top:self.top+rh,self.left:self.left+rw]=cv2.resize(original,(rw,rh))
        self.active=False;self.paused=False;self.target_id=0;self.pass_id=0
        self.posture_label='neutral'
        self.target_started=None;self.valid_count=0;self.row_count=0;self.closed=False
        self.target_seconds=3.0
        self._skip_transition_sample=False
        self.session_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        root=Path(output_root or Path(__file__).parent/'target_records')
        self.folder=root/(self.session_id+'_'+model)
        self.folder.mkdir(parents=True,exist_ok=False)
        self.csv_path=self.folder/'gaze_target_distances.csv'
        self.handle=self.csv_path.open('w',encoding='utf-8-sig',newline='')
        self.writer=csv.DictWriter(self.handle,fieldnames=FIELDS)
        self.writer.writeheader();self.handle.flush()
        self.event_handle=(self.folder/'manual_events.csv').open('w',encoding='utf-8-sig',newline='')
        self.events=csv.writer(self.event_handle)
        self.events.writerow(['timestamp_unix','event','pass_id','target_id'])
        metadata=dict(schema='timed-target-recording-v2',model=model,session_id=self.session_id,
            canvas_size=[self.width,self.height],reference_image_size=[1920,1080],image=str(self.image),
            transform=dict(left=self.left,top=self.top,sx=self.sx,sy=self.sy),
            targets={str(i+1):p for i,p in enumerate(TARGETS)},
            controls='1 neutral, 2 forward, 3 backward, 4 neutral_return (between passes); S start; P pause; E stop',
            ground_truth='instructed target, not measured eye-tracker ground truth',
            error_output='model coordinate supplied by caller; displayed coordinate recorded separately',
            settling='all rows retained; settled_1s=0 for first second after target/resume',
            automatic_target_switching=True,target_seconds=self.target_seconds,
            timing='software frame refresh; samples at age >=3s discarded; no unseen-target catch-up',
            pose_pairing='base and corrected coordinates use the same frame and smoothing when supplied')
        (self.folder/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        atexit.register(self.close)
        print('[TARGET CSV]',self.csv_path)

    def _event(self,name):
        self.events.writerow([time.time(),name,self.pass_id,self.target_id]);self.event_handle.flush()

    def key(self,key):
        """Caller forwards ordinary ASCII key presses after native calibration."""
        if key in (ord('1'),ord('2'),ord('3'),ord('4')) and not self.active:
            self.posture_label={'1':'neutral','2':'forward','3':'backward','4':'neutral_return'}[chr(key)]
            self._event('posture_'+self.posture_label)
        elif key in (ord('s'),ord('S')) and not self.active:
            self.pass_id+=1;self.target_id=1;self.active=True;self.paused=False
            self.target_started=time.monotonic();self._event('start')
            self._skip_transition_sample=True
        elif key in (ord('p'),ord('P')) and self.active:
            self.paused=not self.paused
            if not self.paused:
                self.target_started=time.monotonic()
                self._skip_transition_sample=True
            self._event('pause' if self.paused else 'resume')
        elif key in (ord('e'),ord('E')):
            self.stop('stop')

    def stop(self,reason='stop'):
        if self.active:self._event(reason)
        self.active=False;self.paused=False

    @property
    def status(self):
        state=('PAUSED' if self.paused else 'RECORDING') if self.active else 'READY / STOPPED'
        number=f' | TARGET {self.target_id}/12' if self.active else ''
        left=max(0.,self.target_seconds-(time.monotonic()-self.target_started)) if self.active else 0.
        return f'{state} [{self.posture_label}]{number} {left:.1f}s | 1-4:posture S:start P:pause E:stop'

    def _advance_for_display(self):
        if not self.active or self.paused:return
        if time.monotonic()-self.target_started>=self.target_seconds:
            self._event('auto_timeout')
            if self.target_id==12:
                self.stop('completed_12')
            else:
                self.target_id+=1
                self.target_started=time.monotonic()
                self._skip_transition_sample=True
                self._event('auto_next')

    def frame(self):
        import cv2
        self._advance_for_display()
        canvas=self.background.copy()
        if self.active:
            x,y=TARGETS[self.target_id-1]
            point=(round(self.left+x*self.sx),round(self.top+y*self.sy))
            cv2.circle(canvas,point,max(14,round(43*min(self.sx,self.sy))),(0,140,230),2)
        cv2.putText(canvas,self.status,(12,max(20,self.height-45)),cv2.FONT_HERSHEY_SIMPLEX,
                    max(.35,min(.62,self.width/2100)),(35,35,35),1,cv2.LINE_AA)
        return canvas

    def record(self,point,displayed=None,baseline=None,correction_status='',pose_policy='',pose_delta=None,
               head_point=None,shoulders_point=None,head_status='',shoulders_status=''):
        if not self.active or self.paused or self.closed:return
        age=time.monotonic()-self.target_started
        if age>=self.target_seconds:return
        if self._skip_transition_sample:
            self._skip_transition_sample=False
            return
        xr,yr=TARGETS[self.target_id-1]
        tx,ty=self.left+xr*self.sx,self.top+yr*self.sy
        good=pair(point)
        row=dict.fromkeys(FIELDS,'')
        row.update(model=self.model,session_id=self.session_id,timestamp_unix=time.time(),
            pass_id=self.pass_id,target_id=self.target_id,target_age_s=age,settled_1s=int(age>=1),
            valid=int(good),target_x_px=tx,target_y_px=ty,target_x_image_px=xr,target_y_image_px=yr,
            canvas_width=self.width,canvas_height=self.height,
            pose_correction_status=correction_status,pose_policy=pose_policy,
             posture_label=self.posture_label,head_correction_status=head_status,
             shoulders_correction_status=shoulders_status)
        if good:
            gx,gy=map(float,point);dx,dy=gx-tx,gy-ty
            ix,iy=(gx-self.left)/self.sx,(gy-self.top)/self.sy
            row.update(gaze_x_px=gx,gaze_y_px=gy,dx_px=dx,dy_px=dy,distance_px=math.hypot(dx,dy),
                       gaze_x_image_px=ix,gaze_y_image_px=iy,distance_image_px=math.hypot(ix-xr,iy-yr))
            self.valid_count+=1
        if pair(displayed):row.update(displayed_x_px=float(displayed[0]),displayed_y_px=float(displayed[1]))
        if pair(baseline):
            bx,by=map(float,baseline)
            row.update(base_gaze_x_px=bx,base_gaze_y_px=by,base_distance_px=math.hypot(bx-tx,by-ty),
                       base_distance_image_px=math.hypot((bx-self.left)/self.sx-xr,(by-self.top)/self.sy-yr))
        for name, candidate in [('head',head_point),('shoulders',shoulders_point)]:
            if pair(candidate):
                cx,cy=map(float,candidate)
                row.update({name+'_gaze_x_px':cx,name+'_gaze_y_px':cy,
                            name+'_distance_image_px':math.hypot((cx-self.left)/self.sx-xr,
                                                                (cy-self.top)/self.sy-yr)})
        if pair(pose_delta):row.update(pose_delta_x_px=float(pose_delta[0]),pose_delta_y_px=float(pose_delta[1]))
        self.writer.writerow(row);self.handle.flush();self.row_count+=1

    def close(self):
        if not self.closed:
            self.stop('program_exit')
            self.handle.close();self.event_handle.close();self.closed=True

    def finish(self,show=True):
        self.close()
        if self.valid_count==0:
            print('[TARGET] No valid manually recorded samples; CSV retained:',self.csv_path)
            return None
        import importlib.util
        python=sys.executable
        if importlib.util.find_spec('matplotlib') is None:
            python=os.getenv('GAZE_HEATMAP_PYTHON',str(Path.home()/'miniconda3'/'envs'/'x_gaze'/'python.exe'))
        output=self.folder/'target_heatmap.png'
        command=[python,str(Path(__file__).with_name('render_target_heatmap.py')),
                 '--csv',str(self.csv_path),'--image',str(self.image),'--output',str(output)]
        if show:command.append('--show')
        subprocess.run(command,check=True)
        print('[TARGET HEATMAP]',output)
        return output
