"""RTMPose-only visualization of five non-clinical posture measures."""
import argparse
import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rtmlib import Body, draw_skeleton
from posture_protractor import measure, draw_camera


NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH = 0, 1, 2, 3, 4, 5, 6
FONT_PATH = r'C:\Windows\Fonts\malgun.ttf'


def midpoint(points, scores, a, b, threshold):
    if scores[a] < threshold or scores[b] < threshold:
        return None
    return (points[a] + points[b]) / 2.0


def signed_from_vertical(vector):
    """Image-plane angle: right-positive, vertical-up is zero."""
    return math.degrees(math.atan2(float(vector[0]), float(-vector[1])))


def visible_geometry(points, scores, threshold=.25):
    shoulder = midpoint(points, scores, LSH, RSH, threshold)
    ears = midpoint(points, scores, LEAR, REAR, threshold)
    eyes = midpoint(points, scores, LEYE, REYE, threshold)
    if ears is None and scores[NOSE] >= threshold:
        head = points[NOSE]
    elif ears is not None and scores[NOSE] >= threshold:
        head = .65 * ears + .35 * points[NOSE]
    else:
        head = ears
    if shoulder is None:
        return None
    width=float(np.linalg.norm(points[LSH]-points[RSH]))
    if width < 10:
        return None
    face_pitch=None
    if eyes is not None and scores[NOSE] >= threshold:
        eye_width=float(np.linalg.norm(points[LEYE]-points[REYE]))
        if eye_width >= 3:
            # A within-person image-plane proxy; this is not an anatomical pitch angle.
            face_pitch=math.degrees(math.atan2(float(points[NOSE][1]-eyes[1]),eye_width))
    head_offset=None if head is None else (head-shoulder)/width
    return {'shoulder':shoulder,'shoulder_width':width,'head_offset':head_offset,
            'face_pitch_proxy':face_pitch}


def posture_metrics(points, scores, baseline=None, threshold=.25):
    geometry=visible_geometry(points,scores,threshold)
    values = {'head_bow_change_proxy_deg':None, 'upper_body_forward_change_percent':None,
              'shoulder_tilt_deg':None, 'head_shoulder_relative_change_percent':None}
    shoulder = midpoint(points, scores, LSH, RSH, threshold)
    if shoulder is not None:
        line = points[RSH] - points[LSH]
        angle = math.degrees(math.atan2(float(line[1]),float(line[0])))
        values['shoulder_tilt_deg'] = (angle+90) % 180-90
    if geometry is not None and baseline is not None:
        if geometry['face_pitch_proxy'] is not None and baseline['face_pitch_proxy'] is not None:
            values['head_bow_change_proxy_deg']=(geometry['face_pitch_proxy']-
                                                  baseline['face_pitch_proxy'])
        values['upper_body_forward_change_percent']=(geometry['shoulder_width']/
                                                       baseline['shoulder_width']-1.0)*100.0
        if geometry['head_offset'] is not None and baseline['head_offset'] is not None:
            delta=geometry['head_offset']-baseline['head_offset']
            values['head_shoulder_relative_change_percent']=float(np.linalg.norm(delta)*100.0)
    return values


def normalized_pose(points, scores, threshold=.25):
    shoulder=midpoint(points,scores,LSH,RSH,threshold)
    if shoulder is None:return None,None
    width=np.linalg.norm(points[LSH]-points[RSH])
    if width < 10:return None,None
    ids=(NOSE,LEYE,REYE,LEAR,REAR,LSH,RSH)
    valid=np.array([scores[i]>=threshold for i in ids])
    pose=np.zeros((len(ids),2),float)
    pose[valid]=(points[np.array(ids)[valid]]-shoulder)/width
    return pose,valid


def personal_change(current, baseline):
    if current is None or baseline is None:return None
    pose,valid=current;base_pose,base_valid=baseline
    if any(value is None for value in (pose,valid,base_pose,base_valid)):
        return None
    common=valid & base_valid
    if common.sum()<4:return None
    return float(np.sqrt(np.mean(np.sum((pose[common]-base_pose[common])**2,axis=1)))*100)


def ema(previous,current,alpha=.18):
    if current is None:return previous
    return current if previous is None else alpha*current+(1-alpha)*previous


def put_korean(image, lines):
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    canvas=Image.fromarray(rgb);draw=ImageDraw.Draw(canvas)
    title=ImageFont.truetype(FONT_PATH,25);body=ImageFont.truetype(FONT_PATH,22)
    for text,xy,color,is_title in lines:
        draw.text(xy,text,font=title if is_title else body,fill=color)
    return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)


def fmt(value):return 'N/A' if value is None else f'{value:+6.1f}°'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--threshold',type=float,default=.25)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'rtmpose_sessions')
    args=parser.parse_args()
    estimator=Body(mode='lightweight',backend='onnxruntime',device='cpu')
    cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW)
    if not cap.isOpened():raise RuntimeError(f'Cannot open camera {args.camera}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    args.output.mkdir(parents=True,exist_ok=True)
    session=args.output/datetime.now().strftime('%Y%m%d_%H%M%S');session.mkdir()
    log=(session/'posture_metrics.csv').open('w',newline='',encoding='utf-8-sig')
    fields=('timestamp','head_bow_change_proxy_deg','upper_body_forward_change_percent',
            'shoulder_tilt_deg','head_shoulder_relative_change_percent',
            'personal_change_percent_shoulder_width')
    writer=csv.DictWriter(log,fieldnames=fields);writer.writeheader()
    baseline=None;smoothed={key:None for key in fields[1:-1]};change_smooth=None
    histories={key:deque(maxlen=90) for key in fields[1:]}
    last_log=0.;last_points=None;last_scores=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok:continue
            keypoints,scores=estimator(frame)
            if len(keypoints):
                # Largest detected person by visible keypoint span.
                spans=[np.ptp(k,axis=0).prod() for k in keypoints]
                idx=int(np.argmax(spans));points=np.asarray(keypoints[idx]);confidence=np.asarray(scores[idx])
                last_points,last_scores=points,confidence
                raw=posture_metrics(points,confidence,baseline,args.threshold)
                for name,value in raw.items():smoothed[name]=ema(smoothed[name],value)
                current=normalized_pose(points,confidence,args.threshold)
                change=personal_change(current,None if baseline is None else baseline['pose'])
                change_smooth=ema(change_smooth,change)
                view=draw_skeleton(frame.copy(),points[None],confidence[None],kpt_thr=args.threshold,
                                   openpose_skeleton=False,radius=4,line_width=3)
                view=draw_camera(view,measure(points,confidence,baseline,args.threshold))
            else:
                current=None;change_smooth=None
                view=frame.copy()
            view=cv2.resize(view,(720,540),interpolation=cv2.INTER_AREA)
            panel=np.full((540,500,3),245,np.uint8)
            labels=(('1. 머리 숙임 변화 추정',smoothed['head_bow_change_proxy_deg']),
                    ('2. 상체 전방 이동 추정',smoothed['upper_body_forward_change_percent']),
                    ('3. 어깨 기울기',smoothed['shoulder_tilt_deg']),
                    ('4. 머리-어깨 상대 변화',smoothed['head_shoulder_relative_change_percent']),
                    ('5. 개인 내 자세 변화량',change_smooth))
            lines=[('RTMPose-S 자세 지표',(20,18),(25,70,110),True)]
            for i,(name,value) in enumerate(labels):
                if value is None:
                    shown='N/A'
                elif i == 1:
                    shown=f'{value:6.1f} %'
                elif i in (3,4):
                    shown=f'{value:6.1f} %SW'
                else:
                    shown=fmt(value)
                lines.append((name,(22,85+i*78),(40,40,40),False))
                lines.append((shown,(300,85+i*78),(150,40,80),False))
            status='기준 자세: 설정됨' if baseline is not None else 'B 키를 눌러 개인 기준 자세 설정'
            lines.extend([(status,(20,474),(0,105,30),False),
                          ('Q: 종료  |  B: 기준 재설정',(20,510),(70,70,70),False)])
            panel=put_korean(panel,lines)
            combined=np.hstack((view,panel))
            cv2.putText(combined,'RTMPose-S / visible upper-body change proxies - not spine angles',
                        (12,528),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,220),2,cv2.LINE_AA)
            cv2.imshow('RTMPose posture metrics',combined)
            now=time.monotonic()
            if now-last_log>=.1:
                row={'timestamp':time.time(),**smoothed,
                     'personal_change_percent_shoulder_width':change_smooth}
                writer.writerow(row);log.flush();last_log=now
            key=cv2.waitKey(1)&255
            if key in (ord('q'),27):break
            if key in (ord('b'),ord('B')) and last_points is not None:
                candidate=normalized_pose(last_points,last_scores,args.threshold)
                geometry=visible_geometry(last_points,last_scores,args.threshold)
                if candidate[0] is not None and geometry is not None:
                    baseline={'pose':(candidate[0].copy(),candidate[1].copy()),
                              'shoulder_width':geometry['shoulder_width'],
                              'head_offset':None if geometry['head_offset'] is None else geometry['head_offset'].copy(),
                              'face_pitch_proxy':geometry['face_pitch_proxy']}
                    change_smooth=0.
    finally:
        log.close();cap.release();cv2.destroyAllWindows()
        print('SAVED',session/'posture_metrics.csv')


if __name__=='__main__':main()
