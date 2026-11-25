"""USB webcam gaze calibration with independent target validation."""
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from gaze_mapping import PoseCorrectionModel, ScreenMap, fixation_quality
from gaze_visualization import save_gaze_visualization
from dynamic_gaze import NinePointCandidateCalibration, NinePointDynamicCalibration, head_features, pose_inputs
from target_recording import TargetRecorder
from pose_experiment import PoseExperimentCalibration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', type=int, default=int(os.getenv('CAMERA_INDEX', '0')))
    p.add_argument('--monitor', type=int, default=0)
    p.add_argument('--camera-params', type=Path)
    p.add_argument('--pose-device', choices=['auto','npu','cpu'], default='auto')
    p.add_argument('--calibration-mode', choices=['dynamic', 'legacy'], default='dynamic',
                   help='Both modes use only nine targets; dynamic includes head/shoulder features')
    p.add_argument('--source', type=Path, default=Path(os.getenv('XGAZE_SOURCE', str(Path(__file__).resolve().parent.parent / 'xgaze-js' / 'python'))))
    p.add_argument('--pose-policy', choices=['experiment','validated'], default='experiment',
                   help='experiment fits nine-point pose data and keeps bounded correction active as posture changes')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    if args.calibration_mode != 'dynamic' and args.pose_policy == 'experiment':
        p.error('The experiment policy requires --calibration-mode dynamic; use --pose-policy validated for legacy mode')
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass
    import cv2
    import torch
    import mediapipe as mp
    from screeninfo import get_monitors
    from rtmpose_gaze_overlay import RTMPoseGazeOverlay
    sys.path.insert(1, str(args.source))
    from asset import GazeEstimator, detect_faces_mediapipe
    torch.set_num_threads(2)
    data = args.source.parent/'data'
    checkpoint = data/'finetuned_eth-xgaze_resnet18.pth'
    params = args.camera_params or data/'sample_params.yaml'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    estimator = GazeEstimator(str(checkpoint), str(params), str(data/'eth-xgaze.yaml'))
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    estimator.gaze_estimation_model.load_state_dict(weights['model'], strict=True)
    if args.check:
        print('CHECK OK: imports and strict gaze checkpoint; no camera opened')
        return
    solutions = mp.solutions if hasattr(mp, 'solutions') else __import__('mediapipe.python.solutions', fromlist=['face_mesh'])
    detector = solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=.5, min_tracking_confidence=.5)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f'Cannot open USB camera {args.camera}')
    overlay = None
    target_recorder = None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Camera returned no frame')
        fh, fw = frame.shape[:2]
        camera = estimator.camera
        camera.camera_matrix = np.diag([fw/camera.width, fh/camera.height, 1.])@camera.camera_matrix
        camera.width, camera.height = fw, fh
        monitors = get_monitors()
        monitor = monitors[args.monitor]
        w, h = int(monitor.width), int(monitor.height)
        overlay = RTMPoseGazeOverlay(preferred_device=args.pose_device)
        folder = overlay.session
        target_recorder = TargetRecorder(w, h, 'ours', output_root=folder/'target_records')
        metadata = dict(camera_index=args.camera, resolution=[fw, fh], monitor=[monitor.x, monitor.y,w,h],
                        camera_params=str(params), intrinsics_measured=args.camera_params is not None,
                        pose_backend=overlay.backend, pose_device=overlay.device,
                        effective_camera_matrix=camera.camera_matrix.tolist(),
                        mapping=args.calibration_mode, pose_policy=args.pose_policy,
                        experiment_note="candidate-on evaluation; inspect validation and per-frame status",
                        validation=('same nine fixations; internal CV reported; bounded posture extrapolation' if args.pose_policy == 'experiment'
                                    else 'internal leave-one-target-out CV on initial nine fixations' if args.calibration_mode == 'dynamic'
                                    else 'leave-one-target-out internal cross-validation'),
                        head_features='face center x/z,y/z,log(z), rotation first two columns; model-derived',
                        shoulder_features='seven 2D proxies; maximum age 150ms in dynamic mode')
        (folder/'setup.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        cv2.namedWindow('Gaze', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Gaze', monitor.x, monitor.y)
        cv2.setWindowProperty('Gaze', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        train = [(x,y) for y in (.12,.5,.88) for x in (.12,.5,.88)]
        stage, index, start = 'pose_ready', 0, None
        pose_ready_since = None
        samples, sample_poses, fit_x, fit_y = [], [], [], []
        calibration_trials = []
        dynamic_trials, dynamic_samples = [], []
        model, pose_model, filtered, previous = None, None, None, time.monotonic()
        base_filtered = None
        head_filtered = shoulders_filtered = None
        dynamic = None
        show, notice = True, ''
        attempts = []
        live_points = []
        with (folder/'frames.jsonl').open('w', encoding='utf-8') as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera disconnected')
                tick = time.monotonic()
                pose_interval = (.10 if args.calibration_mode == 'dynamic' else
                                 0.0 if stage == 'pose_ready' else
                                 overlay.calibration_interval if stage == 'train' else
                                 overlay.live_interval)
                view = overlay.update(frame, min_submit_interval=pose_interval)
                pose_state = overlay.pose_vector()
                raw = None
                head_state = None
                try:
                    # The pose readiness step needs only RTMPose; avoid blocking its
                    # display with an unused gaze network forward pass.
                    faces = [] if stage == 'pose_ready' else detect_faces_mediapipe(detector, frame)
                    if len(faces) == 1:
                        with torch.inference_mode():
                            estimator.estimate(faces[0], frame)
                        v = np.asarray(faces[0].gaze_vector).ravel()
                        if np.isfinite(v).all() and abs(v[2]) > .1:
                            raw = v[:2]/v[2]
                            head_state = head_features(faces[0])
                except (ValueError, cv2.error):
                    pass
                canvas = target_recorder.frame() if stage == 'live' else np.full((h,w,3), 245, np.uint8)
                target = None
                display_point = None
                record_point = None
                base_record_point = None
                prediction = None
                base_prediction = None
                head_prediction = shoulders_prediction = None
                head_status = shoulders_status = 'not_available'
                head_record_point = shoulders_record_point = None
                correction = np.zeros(2,float)
                correction_status = 'base'
                # Age relative to the current gaze frame, not inference completion.
                pose_age = None if overlay.result_time is None else tick-overlay.result_time
                inputs = pose_inputs(head_state, pose_state, pose_age)
                if model is not None and raw is not None:
                    base_prediction=model.predict(raw)
                    if dynamic is not None and dynamic.finished:
                        correction, correction_status = dynamic.predict(raw, inputs)
                    elif pose_model is not None:
                        correction=pose_model.predict(pose_state)
                    prediction=base_prediction+correction
                    # Counterfactual ablation: independent candidate outputs on the SAME frame.
                    # These are diagnostic, NOT independently validated operating modes.
                    if dynamic is not None and dynamic.finished:
                        for route in ('head', 'head_shoulders'):
                            component = dynamic.models.get(route)
                            if component is None:
                                continue
                            delta, reason = component.correction(
                                raw, inputs.get(route), candidate=True, extrapolate=True)
                            candidate_pred = base_prediction + delta if reason == 'applied' else None
                            if route == 'head':
                                head_prediction, head_status = candidate_pred, reason
                            else:
                                shoulders_prediction, shoulders_status = candidate_pred, reason
                message = 'POSE CHECK: keep face, eyes, nose, ears and shoulders visible'
                if stage == 'pose_ready':
                    if pose_state is None:
                        pose_ready_since=None
                        notice='Waiting for a complete, stable RTMPose observation'
                    else:
                        if pose_ready_since is None:
                            pose_ready_since=tick
                        remaining=max(0.,1.5-(tick-pose_ready_since))
                        notice=f'Pose detected: calibration starts automatically in {remaining:.1f}s'
                        if tick-pose_ready_since>=1.5 and overlay.set_baseline():
                            stage,start='train',None
                            notice='Automatic posture reference captured'
                elif stage == 'train':
                    targets = train
                    target = np.array(targets[index])
                    if start is None:
                        start = tick
                    message = f'{stage.upper()} {index+1}/{len(targets)}: keep looking at red target (automatic)'
                    cv2.circle(canvas, tuple((target*[w,h]).astype(int)), 12, (0,0,220), -1)
                    if start is not None:
                        elapsed = tick-start
                        _, quality = fixation_quality(samples)
                        spread = '---' if quality['spread'] is None else f"{quality['spread']:.3f}"
                        message = (f'{stage.upper()} collecting {elapsed:.1f}s | valid {len(samples)}/8 '
                                   f'| retained {quality["retained"]} | spread {spread}')
                        if elapsed >= 1:
                            if raw is None:
                                samples = []
                                sample_poses = []
                                dynamic_samples = []
                            else:
                                samples.append(raw.tolist())
                                sample_poses.append(None if pose_state is None else pose_state.tolist())
                                dynamic_samples.append(dict(raw=raw.tolist(), target=target.tolist(), **inputs))
                        ready = elapsed >= 5 and len(samples) >= 8
                        timeout = elapsed >= 10
                        if ready or timeout:
                            accepted, quality = fixation_quality(samples)
                        if (ready and accepted is not None) or timeout:
                            attempts.append(dict(stage=stage,index=index,target=target.tolist(),samples=samples,
                                                 pose_samples=sample_poses,accepted=accepted is not None,quality=quality))
                            (folder/'trials.json').write_text(json.dumps(attempts), encoding='utf-8')
                            if accepted is None:
                                notice = (f'Retry: {quality["reason"]}; valid={quality["samples"]}, '
                                          f'retained={quality["retained"]}, spread={quality["spread"]}')
                            else:
                                notice = ''
                                fit_x.append(accepted); fit_y.append(target.copy())
                                calibration_trials.append({'target':target.tolist(),
                                                           'raw':[list(value) for value in samples],
                                                           'pose':list(sample_poses)})
                                dynamic_trials.append(dict(target=target.tolist(), samples=list(dynamic_samples)))
                                index += 1
                                if index == len(targets):
                                    index = 0
                                    model = ScreenMap().fit(fit_x, fit_y)
                                    (folder/'screen_map.json').write_text(json.dumps(model.as_dict()), encoding='utf-8')
                                    if args.calibration_mode == 'dynamic':
                                        calibrator = NinePointCandidateCalibration if args.pose_policy == 'experiment' else NinePointDynamicCalibration
                                        dynamic = calibrator(model, [w, h]).fit(dynamic_trials)
                                        (folder/'dynamic_trials.json').write_text(json.dumps(dynamic.trials), encoding='utf-8')
                                        (folder/'dynamic_pose_correction.json').write_text(
                                            json.dumps(dynamic.as_dict(), indent=2), encoding='utf-8')
                                        (folder/'validation.json').write_text(json.dumps(dict(
                                            models={k: m.report for k, m in dynamic.models.items()},
                                            routing=dynamic.routing_report), indent=2), encoding='utf-8')
                                        stage = 'review'
                                        notice = '9-point calibration complete | Correction: '+(dynamic.selected or 'OFF')+' | ENTER: live'
                                    else:
                                        pose_model=PoseCorrectionModel().fit(model,calibration_trials)
                                        (folder/'pose_correction.json').write_text(
                                            json.dumps(pose_model.as_dict(),indent=2),encoding='utf-8')
                                        report=pose_model.report
                                        (folder/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                                        notice=f'Pose correction {"ON" if pose_model.enabled else "OFF"} | ENTER: live'
                                        stage='review'
                            start, samples, sample_poses = None, [], []
                            dynamic_samples = []
                    if samples:
                        cloud = np.asarray(samples)
                        center = np.median(cloud, axis=0)
                        cloud_origin = np.array([w-125, h-125])
                        cv2.circle(canvas, tuple(cloud_origin), 90, (190,190,190), 1)
                        for sample in cloud[-40:]:
                            delta = np.clip((sample-center)/.18, -1, 1)
                            dot = (cloud_origin + delta*np.array([85,85])).astype(int)
                            cv2.circle(canvas, tuple(dot), 3, (0,130,210), -1)
                        cv2.putText(canvas, 'FIXATION CLOUD', (w-240,h-225),
                                    cv2.FONT_HERSHEY_SIMPLEX,.45,(70,70,70),1)
                elif stage == 'pose_adapt':
                    target, message, notice = dynamic.update(tick, raw, inputs)
                    notice += ' | head '+('OK' if inputs['head'] is not None else 'MISSING')+' | shoulders '+('FRESH' if inputs['head_shoulders'] is not None else 'MISSING/STALE')
                    cv2.circle(canvas, tuple((target*[w,h]).astype(int)), 12, (0,0,220), -1)
                    if dynamic.finished:
                        (folder/'pose_adaptation_trials.json').write_text(json.dumps(dynamic.trials), encoding='utf-8')
                        validation_report = dict(models={k:m.report for k,m in dynamic.models.items()},
                                                 routing=dynamic.routing_report,
                                                 note='Separate posture validation; candidate-on evaluation can override a failed validation gate')
                        (folder/'validation.json').write_text(json.dumps(validation_report,indent=2), encoding='utf-8')
                        try:
                            if args.pose_policy == 'experiment':
                                routes = dynamic.enable_experiment()
                                notice = 'EXPERIMENT correction ON: '+','.join(routes)+' | ENTER: target evaluation'
                            else:
                                routes = [k for k, m in dynamic.models.items() if m.enabled]
                                notice = 'Validated correction: '+(','.join(routes) or 'OFF')+' | ENTER: target evaluation'
                            stage = 'review'
                        except ValueError as error:
                            notice = str(error)
                            stage = 'pose_failed'
                        (folder/'dynamic_pose_correction.json').write_text(json.dumps(dynamic.as_dict(),indent=2), encoding='utf-8')
                elif stage == 'pose_failed':
                    message = 'No fitted posture model. Keep face/shoulders visible and press R to repeat calibration.'
                elif stage == 'review':
                    message = 'Calibration complete; validation results saved'
                elif stage == 'live':
                    message = 'LIVE | V: posture view | R: repeat posture + gaze calibration | Q: quit'
                    if dynamic is not None:
                        notice = 'Correction: '+correction_status
                    if prediction is not None:
                        dt = min(tick-previous, .25)
                        alpha = 1-np.exp(-dt/.045)
                        filtered = prediction.copy() if filtered is None else filtered+alpha*(prediction-filtered)
                        base_filtered = base_prediction.copy() if base_filtered is None else base_filtered+alpha*(base_prediction-base_filtered)
                        base_record_point = base_filtered*[w,h]
                        if head_prediction is not None:
                            head_filtered = head_prediction.copy() if head_filtered is None else head_filtered+alpha*(head_prediction-head_filtered)
                            head_record_point = head_filtered*[w,h]
                        else:
                            head_filtered = None
                        if shoulders_prediction is not None:
                            shoulders_filtered = shoulders_prediction.copy() if shoulders_filtered is None else shoulders_filtered+alpha*(shoulders_prediction-shoulders_filtered)
                            shoulders_record_point = shoulders_filtered*[w,h]
                        else:
                            shoulders_filtered = None
                        record_point = filtered*[w,h]
                        point = np.clip(record_point, [0,0], [w-1,h-1]).astype(int)
                        display_point = point
                        live_points.append((time.time(),int(point[0]),int(point[1])))
                        cv2.circle(canvas, tuple(point), 9, (220,80,0), -1)
                    else:
                        filtered = None
                        base_filtered = None
                        head_filtered = shoulders_filtered = None
                if stage == 'pose_ready' or (stage == 'live' and show):
                    vw = min(w-40, 800 if stage == 'pose_ready' else 500)
                    vh = min(h-150, int(view.shape[0]*vw/view.shape[1]))
                    canvas[110:110+vh,w-vw-20:w-20] = cv2.resize(view,(vw,vh))
                face_text = ('RTMPose automatic pose reference' if stage == 'pose_ready' else
                             'FACE OK' if raw is not None else 'FACE / GAZE NOT DETECTED')
                face_color = (0,125,30) if raw is not None else (0,0,200)
                cv2.putText(canvas,message,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.6,(35,35,35),2)
                cv2.putText(canvas,notice,(20,70),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,180),1)
                cv2.putText(canvas,face_text,(20,100),cv2.FONT_HERSHEY_SIMPLEX,.5,face_color,2)
                compute_ms = (time.monotonic()-tick)*1000
                cv2.putText(canvas, f'loop {compute_ms:.0f} ms | pose age '
                            f'{"---" if overlay.result_time is None else f"{time.monotonic()-overlay.result_time:.1f}s"}',
                            (20,h-20),cv2.FONT_HERSHEY_SIMPLEX,.48,(80,80,80),1)
                if stage == 'live':
                    target_recorder.record(record_point, displayed=display_point, baseline=base_record_point,
                                           correction_status=correction_status, pose_policy=args.pose_policy,
                                           pose_delta=correction*np.array([w,h]),
                                           head_point=head_record_point, shoulders_point=shoulders_record_point,
                                           head_status=head_status, shoulders_status=shoulders_status)
                    overlay.log(None if display_point is None else int(display_point[0]),
                                None if display_point is None else int(display_point[1]))
                log.write(json.dumps(dict(timestamp=time.time(),stage=stage,target=None if target is None else target.tolist(),raw=None if raw is None else raw.tolist(),prediction=None if prediction is None else prediction.tolist(),
                    base_prediction=None if base_prediction is None else base_prediction.tolist(),
                    correction=correction.tolist(), correction_status=correction_status,
                    posture_label=target_recorder.posture_label if stage == 'live' else None,
                    head_prediction=None if head_prediction is None else head_prediction.tolist(),
                    shoulders_prediction=None if shoulders_prediction is None else shoulders_prediction.tolist(),
                    head_status=head_status, shoulders_status=shoulders_status,
                    pose_inputs=inputs, processing_ms=compute_ms,pose_age_s=pose_age))+'\n')
                previous = tick
                cv2.imshow('Gaze',canvas)
                key = cv2.waitKey(1)&255
                if stage == 'live': target_recorder.key(key)
                if key in (ord('q'),ord('Q'),27): break
                if key == 32 and stage == 'pose_adapt': dynamic.confirm()
                if key == 13 and stage == 'review':
                    stage='live'
                    show=False  # Keep the posture inset from covering experiment targets.
                if key == ord('v'): show=not show
                if key == ord('r'):
                    target_recorder.stop('recalibration')
                    stage,index,start='pose_ready',0,None
                    pose_ready_since=None
                    samples,sample_poses,fit_x,fit_y=[],[],[],[]
                    calibration_trials=[]
                    dynamic_trials, dynamic_samples = [], []
                    model,pose_model,filtered=None,None,None
                    base_filtered=None
                    head_filtered = shoulders_filtered = None
                    dynamic = None
                    overlay.clear_baseline()
                    live_points=[]
                    notice=''
        output = target_recorder.finish(show=False)
        if output is not None:
            result=cv2.imdecode(np.fromfile(str(output),dtype=np.uint8),cv2.IMREAD_COLOR)
            cv2.imshow('Gaze',result)
            cv2.waitKey(300)
            while cv2.getWindowProperty('Gaze',cv2.WND_PROP_VISIBLE)>=1:
                result_key=cv2.waitKey(50)&255
                if result_key in (ord('q'),ord('Q'),27,13,32): break
        else:
            print('No valid manually recorded samples. Press S in live view to start target 1.')
        print('TARGET CSV', target_recorder.csv_path)
        print('SESSION',folder)
    finally:
        if target_recorder is not None: target_recorder.close()
        cap.release(); detector.close()
        if overlay is not None: overlay.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
