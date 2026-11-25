"""Personal gaze calibration with 3D pose; run --help without opening a webcam."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from collections import deque
import numpy as np
import cv2
from pose_core import PoseGazeModel, pose_features, aggregate_trial, evaluate_rows, json_default


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


POSE_CONNECTIONS = ((0,11),(0,12),(11,12),(11,13),(13,15),(12,14),(14,16),
                    (11,23),(12,24),(23,24))


def draw_pose_overlay(frame, pose_result, threshold=.35):
    """Draw the pose estimator actually used by the embedding pipeline."""
    output = frame.copy()
    if not pose_result.pose_landmarks:
        cv2.putText(output,'MediaPipe Pose: not detected',(15,30),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2,cv2.LINE_AA)
        return output
    points = pose_result.pose_landmarks.landmark
    h,w=output.shape[:2]
    def xy(i):return int(points[i].x*w),int(points[i].y*h)
    for a,b in POSE_CONNECTIONS:
        if points[a].visibility>=threshold and points[b].visibility>=threshold:
            cv2.line(output,xy(a),xy(b),(0,220,120),3,cv2.LINE_AA)
    for i in sorted({i for edge in POSE_CONNECTIONS for i in edge}):
        if points[i].visibility>=threshold:
            cv2.circle(output,xy(i),5,(0,180,255),-1,cv2.LINE_AA)
    cv2.putText(output,'MediaPipe Pose | inferred 3D features',(15,30),
                cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,120),2,cv2.LINE_AA)
    return output


def embedding_panel(pred, history, width=500, height=360):
    """Show the first 3 PCA components; this is not a metric 3D skeleton."""
    panel=np.full((height,width,3),245,np.uint8)
    cv2.putText(panel,'POSTURE EMBEDDING (PCA 1-3)',(18,30),
                cv2.FONT_HERSHEY_SIMPLEX,.65,(45,45,45),2,cv2.LINE_AA)
    origin=np.array([width//2,height//2+35],dtype=float)
    axes=((np.array([105.,0.]),(0,0,220),'PC1'),
          (np.array([-58.,-82.]),(0,150,0),'PC2'),
          (np.array([58.,-82.]),(220,80,0),'PC3'))
    for vec,color,label in axes:
        end=(origin+vec).astype(int)
        cv2.arrowedLine(panel,tuple(origin.astype(int)),tuple(end),color,2,cv2.LINE_AA,tipLength=.1)
        cv2.putText(panel,label,tuple(end+(5,-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    if pred and pred.get('embedding') is not None:
        z=np.asarray(pred['embedding'],dtype=float)
        if len(z)>=3 and np.isfinite(z[:3]).all():
            history.append(z[:3].copy())
    def project(z):
        z=np.tanh(np.asarray(z)/3.0)
        return (origin+z[0]*axes[0][0]+z[1]*axes[1][0]+z[2]*axes[2][0]).astype(int)
    pts=[project(z) for z in history]
    for i in range(1,len(pts)):
        shade=int(80+150*i/max(1,len(pts)-1))
        cv2.line(panel,tuple(pts[i-1]),tuple(pts[i]),(shade,70,255-shade//2),2,cv2.LINE_AA)
    if pts:cv2.circle(panel,tuple(pts[-1]),8,(180,0,220),-1,cv2.LINE_AA)
    profile=pred.get('profile','unknown') if pred else 'unknown'
    reason=pred.get('reason','no model') if pred else 'no model'
    cv2.putText(panel,f'Auto profile: {profile}',(18,height-48),
                cv2.FONT_HERSHEY_SIMPLEX,.6,(45,45,45),2,cv2.LINE_AA)
    cv2.putText(panel,f'Fusion: {reason}',(18,height-20),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(75,75,75),1,cv2.LINE_AA)
    return panel


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['train','evaluate','live','report','check'], default='check')
    p.add_argument('--source', type=Path, default=Path(r'C:\Users\alsdn\OneDrive\Desktop\xgaze-js\python'))
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'sessions')
    p.add_argument('--participant', default='P001')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--model', type=Path)
    p.add_argument('--session', type=Path, help='Session folder for --mode report')
    p.add_argument('--postures', nargs='+', default=['natural'], help=argparse.SUPPRESS)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--settle', type=float, default=.8)
    p.add_argument('--dwell', type=float, default=1.5)
    p.add_argument('--seed', type=int, default=20260907)
    p.add_argument('--camera-params', type=Path)
    args = p.parse_args()
    if args.mode in ('evaluate','live') and not args.model:
        p.error('--model is required for evaluate/live')
    if args.mode == 'report' and not args.session:
        p.error('--session is required for report')
    if args.repeats < 1 or args.settle < .5 or args.dwell < 1:
        p.error('repeats>=1, settle>=0.5, dwell>=1 are required')
    return args


def make_report(folder):
    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    rows = []
    with (folder/'frames.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    report = evaluate_rows(rows, *meta['screen_size'])
    report['participant'] = meta['participant']
    report['session_id'] = meta['session_id']
    status_path = folder/'completion.json'
    report['completion'] = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'complete':False}
    report['captured_frames'] = len(rows)
    report['valid_gaze_frames'] = sum(r.get('raw') is not None for r in rows)
    report['valid_pose_frames'] = sum(r.get('pose') is not None for r in rows)
    latency = [r['processing_ms'] for r in rows if r.get('processing_ms') is not None]
    report['processing_latency_ms'] = {'median':float(np.median(latency)), 'p95':float(np.quantile(latency,.95))} if latency else None
    # Durations use observed timestamps, capped to avoid attributing gaps to a pose.
    profiles = {}
    for a,b in zip(rows,rows[1:]):
        name = a.get('profile','unknown')
        profiles[name] = profiles.get(name,0)+min(.2,max(0,b['timestamp']-a['timestamp']))
    report['observed_profile_seconds_capped'] = profiles
    report['interpretation'] = 'Profile labels are similarity to calibration poses, not posture quality, comfort inference or concentration.'
    dump(folder/'report.json',report)
    o = report['overall']
    def fmt(v):
        return '미측정' if v is None else f'{v:.2f}'
    md = [f'# 시선·자세 실험 기록: {meta["session_id"]}', '',
          f'참가자: {meta["participant"]} / 모드: {meta["mode"]}', '',
          '이 보고서는 실제 저장된 프레임으로 계산한다. 화면 목표점은 응시 지시 위치이며 실제 안구 측정 정답은 아니다.', '',
          '|항목|값|','|---|---|',
          f'|평가 시도 / 채택 목표 구간|{o["total_trials"]} / {o["accepted_trials"]}|',
          f'|기준 시선 평균 오차(px)|{fmt(o.get("baseline_mean_px"))}|',
          f'|자세 보정 평균 오차(px)|{fmt(o.get("fused_mean_px"))}|',
          f'|오차 감소(px, 음수는 악화)|{fmt(o.get("improvement_px"))}|', '',
          '기준 방식은 동일한 얼굴 시선 모델에 새 수집 절차의 2차 다항 보정을 적용한 통제 비교군이다. 원본 프로그램 전체와 동일한 비교군은 아니다.', '',
          '개인 보정 후 별도 세션 평가이며, 미참여자 일반화·집중도·3D 복원 정확도를 검증한 결과가 아니다. 중단 여부와 자세별 오차는 report.json을 확인한다.']
    (folder/'report.md').write_text('\n'.join(md),encoding='utf-8')
    print('REPORT',folder/'report.md')


def setup(args):
    import cv2
    import torch
    import mediapipe as mp
    sys.path.insert(0,str(args.source.resolve()))
    from asset import GazeEstimator, detect_faces_mediapipe
    data = args.source.parent/'data'
    checkpoint = data/'finetuned_eth-xgaze_resnet18.pth'
    params = args.camera_params or data/'sample_params.yaml'
    normalized = data/'eth-xgaze.yaml'
    for path in (checkpoint,params,normalized):
        if not path.is_file():
            raise FileNotFoundError(path)
    estimator = GazeEstimator(str(checkpoint),str(params),str(normalized),model_name='resnet18')
    # Legacy constructor swallows checkpoint errors: validate strictly ourselves.
    weights = torch.load(str(checkpoint),map_location=estimator.device,weights_only=True)
    estimator.gaze_estimation_model.load_state_dict(weights['model'],strict=True)
    estimator.gaze_estimation_model.eval()
    solutions = mp.solutions if hasattr(mp,'solutions') else __import__('mediapipe.python.solutions',fromlist=['face_mesh','pose'])
    detector = solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=False,
                                          min_detection_confidence=.5,min_tracking_confidence=.5)
    pose = solutions.pose.Pose(model_complexity=1,smooth_landmarks=True,
                               min_detection_confidence=.5,min_tracking_confidence=.5)
    provenance = {'checkpoint_sha256':sha256(checkpoint),'camera_params_sha256':sha256(params),
                  'camera_params_path':str(params.resolve()),'normalized_params_sha256':sha256(normalized),
                  'source_sha256':{p:sha256(args.source/p) for p in ('asset.py','face_cam.py')},
                  'versions':{p:importlib.metadata.version(p) for p in ('numpy','torch','mediapipe','scikit-learn')},
                  'mediapipe_import_version':mp.__version__,
                  'pose_model':'MediaPipe Solutions Pose model_complexity=1',
                  'camera_intrinsics':'scaled provided YAML; sample YAML is not a measured camera calibration'}
    return cv2,torch,estimator,detector,pose,detect_faces_mediapipe,provenance


def main():
    args = arguments()
    if args.mode=='report':
        make_report(args.session)
        return
    cv2,torch,estimator,detector,pose,detect,provenance = setup(args)
    if args.mode=='check':
        # Exercise both landmark graphs on a blank image without activating hardware.
        blank = np.zeros((480,640,3),dtype=np.uint8)
        detector.process(blank)
        pose.process(blank)
        detector.close(); pose.close()
        print('CHECK OK: checkpoint, imports, face/pose graphs. No webcam opened.')
        print(json.dumps(provenance,indent=2))
        return
    model = PoseGazeModel.load(args.model) if args.model else None
    cap = None
    log = None
    folder = None
    complete = False
    trial_records = []
    try:
        actual_camera = None
        requested = [args.camera] + [i for i in range(6) if i != args.camera]
        for camera_index in requested:
            for backend in (cv2.CAP_DSHOW,cv2.CAP_MSMF,cv2.CAP_ANY):
                candidate = cv2.VideoCapture(camera_index,backend)
                if not candidate.isOpened():
                    candidate.release()
                    continue
                candidate.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
                candidate.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
                probe_ok = False
                for _ in range(5):
                    probe_ok, probe_frame = candidate.read()
                    if probe_ok and probe_frame is not None:
                        break
                if probe_ok and probe_frame is not None:
                    cap = candidate
                    actual_camera = camera_index
                    print(f'[INFO] Camera {actual_camera} selected '
                          f'(requested {args.camera}, backend {backend}).')
                    break
                candidate.release()
            if cap is not None:
                break
        if cap is None:
            raise RuntimeError('Cannot acquire frames from camera indexes 0-5. '
                               'Close apps using the camera and check Windows camera permission.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        ok,frame=cap.read()
        if not ok:
            raise RuntimeError('Camera opened but frame acquisition failed')
        fh,fw=frame.shape[:2]
        original_k=estimator.camera.camera_matrix.copy()
        sx,sy=fw/estimator.camera.width,fh/estimator.camera.height
        estimator.camera.camera_matrix=np.diag([sx,sy,1.0])@original_k
        estimator.camera.width,estimator.camera.height=fw,fh
        from screeninfo import get_monitors
        monitors=get_monitors(); monitor=next((m for m in monitors if m.is_primary),monitors[0])
        width,height=int(monitor.width),int(monitor.height)
        cv2.namedWindow('Pose gaze study',cv2.WINDOW_NORMAL)
        cv2.moveWindow('Pose gaze study',monitor.x,monitor.y)
        cv2.setWindowProperty('Pose gaze study',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Pose gaze study',np.full((height,width,3),245,np.uint8));cv2.waitKey(1)
        _,_,ww,hh=cv2.getWindowImageRect('Pose gaze study')
        if ww>0 and hh>0:width,height=ww,hh
        metadata=dict(provenance,participant=args.participant,mode=args.mode,
                      session_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'),
                      screen_size=[width,height],frame_size=[fw,fh],camera_index=actual_camera,
                      requested_camera_index=args.camera,
                      effective_camera_matrix=estimator.camera.camera_matrix.tolist(),
                      settle_seconds=args.settle,dwell_seconds=args.dwell,repeats=args.repeats,
                      seed=args.seed,postures=args.postures,embedding='training-only PCA, at most 8 dimensions',
                      baseline='2nd-order ridge polynomial on raw face-model ray-plane coordinates',
                      model_sha256=sha256(args.model) if args.model else None,
                      study_code_sha256={p:sha256(Path(__file__).resolve().parent/p) for p in ('pose_study.py','pose_core.py')},
                      evaluation='separate session, instructed targets, frozen model')
        if model:
            old=model.data['metadata']
            for key in ('participant','screen_size','frame_size','camera_index','checkpoint_sha256',
                        'camera_params_sha256','normalized_params_sha256','effective_camera_matrix','source_sha256'):
                if old[key]!=metadata[key]:
                    raise ValueError(f'Model/session mismatch: {key}. Recalibrate after changing participant or setup.')
            # UI, logging and visualization edits in pose_study.py do not alter
            # the learned feature vector. Only pose_core.py defines model math.
            old_feature_hash=old.get('feature_code_sha256')
            if old_feature_hash is None:
                old_feature_hash=old.get('study_code_sha256',{}).get('pose_core.py')
            current_feature_hash=metadata['study_code_sha256']['pose_core.py']
            if old_feature_hash != current_feature_hash:
                raise ValueError('Model/session mismatch: pose feature code changed. Recalibrate.')
        folder=args.output/metadata['session_id'];folder.mkdir(parents=True,exist_ok=False)
        dump(folder/'metadata.json',metadata)
        log=(folder/'frames.jsonl').open('w',encoding='utf-8',buffering=1)
        print('SESSION',folder)
        print('Posture is estimated automatically. Sit naturally; change position between blocks if comfortable. SPACE=start, Q=quit.')
        levels=(.12,.5,.88) if args.mode=='train' else (.22,.5,.78)
        rng=np.random.default_rng(args.seed+(0 if args.mode=='train' else 1))
        schedule=[]
        for rep in range(args.repeats * 3):
            for posture_name in ('automatic',):
                points=[(x,y) for x in levels for y in levels]
                rng.shuffle(points)
                schedule.append((str(posture_name),points,rep))
        block=0; point=0; waiting=True; comfort='unknown';start=None;buffer=[];failures=0
        frame_id=0; last_log=0.; rejection_count=0
        embedding_history=deque(maxlen=90)
        while True:
            began=time.perf_counter()
            ok,frame=cap.read(); captured=time.perf_counter()
            if not ok:
                failures+=1
                if failures>30:raise RuntimeError('30 consecutive camera read failures')
                if cv2.waitKey(30)&255 in (27,ord('q')):break
                continue
            failures=0
            if frame.shape[:2]!=(fh,fw):raise RuntimeError('Camera resolution changed; restart and recalibrate')
            raw=None;feature=None;pred=None;reason='no_face'
            faces=detect(detector,frame)
            result=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if len(faces)==1:
                face=faces[0]
                lm=face.landmarks
                def ear(a,b,c,d):
                    return np.linalg.norm(lm[a]-lm[b])/max(np.linalg.norm(lm[c]-lm[d]),1e-6)
                eyes_open=min(ear(159,145,33,133),ear(386,374,362,263))>=.12
                if eyes_open:
                    with torch.inference_mode():estimator.estimate(face,frame)
                    v=np.asarray(face.gaze_vector).ravel();center=np.asarray(face.center).ravel()
                    ray_t=-center[2]/v[2] if abs(v[2])>1e-6 else -1
                    if ray_t>0 and np.isfinite(v).all() and np.isfinite(center).all():
                        raw=(center+ray_t*v)[:2]*1000
                        reason='valid'
                        if result.pose_world_landmarks and result.pose_landmarks:
                            world=np.array([[p.x,p.y,p.z] for p in result.pose_world_landmarks.landmark])
                            img=np.array([[p.x,p.y,p.z,p.visibility] for p in result.pose_landmarks.landmark])
                            feature=pose_features(world,img,face.head_pose_rot.as_matrix(),center)
                        if model:pred=model.predict(raw,feature)
                else:reason='eyes_closed_or_unreliable'
            frame_id+=1
            elapsed=0 if start is None else captured-start
            live=args.mode=='live'
            label,points,rep=schedule[block]
            target=None if live or waiting else points[point]
            sampling=live or (not waiting and elapsed>=args.settle)
            row=dict(frame_id=frame_id,timestamp=captured,phase=args.mode,
                     trial_id=f'{block}:{point}',posture=label if not live else 'unprompted',
                     comfort_self_report=comfort if not live else 'unknown',
                     target=target,raw=raw,pose=feature,gaze_status=reason,
                     baseline=pred['baseline'] if pred else None,fused=pred['fused'] if pred else None,
                     embedding=pred['embedding'] if pred else None,
                     profile=pred['profile'] if pred else 'unknown',
                     fusion_used=pred['fusion_used'] if pred else False,
                     fusion_reason=pred['reason'] if pred else 'not_fitted',
                     processing_ms=(time.perf_counter()-captured)*1000,
                     acquisition_and_processing_ms=(time.perf_counter()-began)*1000)
            if sampling and captured-last_log>=.09:
                log.write(json.dumps(row,default=json_default)+'\n');last_log=captured
                if not live:buffer.append(row)
            canvas=np.full((height,width,3),245,np.uint8)
            if target is not None:
                cv2.circle(canvas,(int(target[0]*width),int(target[1]*height)),12,(0,0,220),-1)
            # Evaluation hides predicted dots to avoid steering the participant.
            if live and pred:
                for key,color in (('baseline',(180,180,180)),('fused',(220,80,0))):
                    xy=pred[key]
                    if np.isfinite(xy).all() and ((xy>=0)&(xy<=1)).all():
                        cv2.circle(canvas,tuple((xy*[width,height]).astype(int)),9,color,-1)
            text1=f'{args.mode.upper()} | Gaze: {reason} | Pose: {"valid" if feature is not None else "missing"}'
            text2=(f'Pose {label}, block {block+1}/{len(schedule)}, target {point+1}/9 | Comfort report: {comfort}'
                   if not live else f'Profile: {row["profile"]} | Fusion: {row["fusion_reason"]}')
            text3='Sit naturally; SPACE: start block | Q: quit' if waiting and not live else 'Look at the RED target and sit naturally. Q: quit'
            for i,line in enumerate((text1,text2,text3)):
                cv2.putText(canvas,line,(20,32+i*30),cv2.FONT_HERSHEY_SIMPLEX,.65,(50,50,50),1)
            if waiting and not live:
                headline = 'POSTURE IS ESTIMATED AUTOMATICALLY'
                detail = f'BLOCK {block+1} OF {len(schedule)}  |  9 RED TARGETS'
                action = 'Sit naturally, then press SPACE'
                for line,y,scale,color in ((headline,height//2-65,1.35,(20,80,160)),
                                           (detail,height//2,0.9,(50,50,50)),
                                           (action,height//2+65,0.85,(0,120,0))):
                    size=cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,scale,2)[0]
                    cv2.putText(canvas,line,((width-size[0])//2,y),cv2.FONT_HERSHEY_SIMPLEX,
                                scale,color,2,cv2.LINE_AA)
            elif not live:
                remaining=max(0.0,args.settle+args.dwell-elapsed)
                progress=f'Target {point+1}/9  |  next in {remaining:.1f}s'
                size=cv2.getTextSize(progress,cv2.FONT_HERSHEY_SIMPLEX,.8,2)[0]
                cv2.putText(canvas,progress,((width-size[0])//2,height-35),
                            cv2.FONT_HERSHEY_SIMPLEX,.8,(60,60,60),2,cv2.LINE_AA)
            cv2.imshow('Pose gaze study',canvas)
            # Preview only while preparing / live: no extra attention target during trials.
            if waiting or live:
                pose_view=draw_pose_overlay(frame,result)
                pose_view=cv2.resize(pose_view,(640,360),interpolation=cv2.INTER_AREA)
                if live:
                    preview=np.hstack((pose_view,embedding_panel(pred,embedding_history)))
                    cv2.imshow('Pose and embedding',preview)
                else:
                    cv2.imshow('Camera alignment',pose_view)
            key=cv2.waitKey(1)&255
            if key in (27,ord('q')):break
            if waiting and not live:
                if key==32:
                    waiting=False;start=time.perf_counter();buffer=[]
                    try:cv2.destroyWindow('Camera alignment')
                    except cv2.error:pass
            if not live and not waiting and elapsed>=args.settle+args.dwell:
                trial=aggregate_trial(buffer,args.mode,f'{block}:{point}',label,comfort,points[point])
                if trial is None:
                    rejection_count+=1
                    print('REJECTED trial',block,point,'insufficient valid gaze; retained in logs')
                else:trial_records.append(trial)
                point+=1;buffer=[];start=time.perf_counter()
                if point==len(points):
                    point=0;block+=1;waiting=True;comfort='unknown';start=None
                    if block==len(schedule):complete=True;break
        if args.mode=='train' and complete:
            try:
                trained=PoseGazeModel.fit(trial_records,metadata)
            except ValueError as error:
                complete=False
                print('[TRAINING INCOMPLETE]',error)
                print('Move the camera back so face and both shoulders remain visible, then repeat training.')
                trained=None
            if trained is not None:
                trained.save(folder/'model.json');print('MODEL',folder/'model.json')
            done=np.full((height,width,3),245,np.uint8)
            lines=(('TRAINING COMPLETE',f'{len(trial_records)} target trials accepted',
                    'model.json saved - press any key') if trained is not None else
                   ('POSTURE DATA INSUFFICIENT','Move camera back: keep face and both shoulders visible',
                    'No model saved - press any key'))
            for line,y,scale,color in ((lines[0],height//2-65,1.3,(0,130,0)),
                                       (lines[1],height//2,0.85,(50,50,50)),
                                       (lines[2],height//2+65,0.75,(50,50,50))):
                size=cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,scale,2)[0]
                cv2.putText(done,line,((width-size[0])//2,y),cv2.FONT_HERSHEY_SIMPLEX,
                            scale,color,2,cv2.LINE_AA)
            cv2.imshow('Pose gaze study',done);cv2.waitKey(0)
        elif args.mode=='train':print('Training interrupted: no model fitted. Raw data retained.')
    finally:
        if log:log.close()
        if cap:cap.release()
        detector.close();pose.close();cv2.destroyAllWindows()
        if folder:
            dump(folder/'trials.json',trial_records)
            dump(folder/'completion.json',{'complete':complete,'mode':args.mode,'accepted_trials':len(trial_records)})
            make_report(folder)


if __name__=='__main__':
    main()
