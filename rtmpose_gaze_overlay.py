"""Asynchronous RTMPose overlay and synchronized gaze/posture logging."""
import csv
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton
from pose_stabilizer import PoseStabilizer
from posture_protractor import measure, draw_camera, draw_panel

from rtmpose_posture_live import (normalized_pose, personal_change, posture_metrics,
                                  visible_geometry, put_korean)


class RTMPoseGazeOverlay:
    def __init__(self, output_root=None, threshold=.4, preferred_device='auto'):
        self.threshold = threshold
        self.stabilizer = PoseStabilizer(threshold)
        self.result_frame = None
        self.submitted_frame = None
        self.backend = 'onnxruntime'
        self.device = 'cpu'
        self.accelerated = False
        if preferred_device in ('auto', 'npu'):
            model_dir = Path(__file__).resolve().parent/'models'/'rtmpose_npu'
            pose_model = model_dir/'rtmpose_s_static.xml'
            detector_model = (Path.home()/'.cache'/'rtmlib'/'hub'/'checkpoints'/
                              'yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx')
            try:
                from openvino import Core
                if 'NPU' not in Core().available_devices:
                    raise RuntimeError('OpenVINO does not report an NPU')
                if not pose_model.is_file() or not detector_model.is_file():
                    raise FileNotFoundError('Static RTMPose NPU model is missing')
                from hybrid_pose import HybridPose
                self.estimator = HybridPose()
                self.backend, self.device, self.accelerated = 'hybrid', 'NPU detector + CPU pose', True
            except Exception as error:
                print('[WARN] NPU initialization failed; using lightweight CPU RTMPose:', error)
                self.estimator = Body(mode='lightweight', backend='onnxruntime', device='cpu')
        else:
            self.estimator = Body(mode='lightweight', backend='onnxruntime', device='cpu')
        self.calibration_interval = .35 if self.accelerated else 2.5
        self.live_interval = .10 if self.accelerated else .75
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rtmpose')
        self.future = None
        self.result_time = None
        self.submitted_time = None
        self.last_submit_time = -1e9
        self.points = None
        self.scores = None
        self.baseline = None
        self.embedding = None
        self.history = deque(maxlen=120)
        self.metrics = None
        root = Path(output_root or Path(__file__).resolve().parent / 'integrated_sessions')
        self.session = root / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session.mkdir(parents=True, exist_ok=True)
        self.handle = (self.session / 'gaze_posture_embedding.csv').open(
            'w', newline='', encoding='utf-8-sig')
        self.fields = (
            'timestamp', 'gaze_x_px', 'gaze_y_px',
            'embedding_lateral_percent_sw', 'embedding_vertical_percent_sw',
            'embedding_depth_proxy_percent', 'head_bow_change_proxy_deg',
            'upper_body_forward_change_percent', 'shoulder_tilt_deg',
            'head_shoulder_relative_change_percent',
            'personal_change_percent_shoulder_width')
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
        self.writer.writeheader()
        self.last_log = 0.0
        print('[INFO] Integrated gaze/posture session:', self.session)
        print(f'[INFO] RTMPose execution: {self.backend}/{self.device}')

    def _infer(self, frame):
        return self.estimator(frame)

    def _consume_result(self):
        if self.future is None or not self.future.done():
            return False
        try:
            keypoints, scores = self.future.result()
            self.result_time = self.submitted_time
            self.result_frame = self.submitted_frame
            self.points = self.scores = None
            if len(keypoints):
                spans = [np.ptp(k, axis=0).prod() for k in keypoints]
                index = int(np.argmax(spans))
                self.points, self.scores = self.stabilizer.update(
                    keypoints[index], scores[index], self.result_time, self.result_frame.shape)
        except Exception as error:
            self.points = self.scores = None
            print('[WARN] RTMPose inference failed:', error)
        self.future = None
        return True

    def _update_values(self):
        if self.points is None:
            self.metrics = None
            self.embedding = None
            return
        self.metrics = posture_metrics(self.points, self.scores, self.baseline, self.threshold)
        current = normalized_pose(self.points, self.scores, self.threshold)
        self.metrics['personal_change_percent_shoulder_width'] = personal_change(
            current, None if self.baseline is None else self.baseline['pose'])
        geometry = visible_geometry(self.points, self.scores, self.threshold)
        if geometry is None or self.baseline is None or geometry['head_offset'] is None:
            self.embedding = None
            return
        base_head = self.baseline['head_offset']
        if base_head is None:
            self.embedding = None
            return
        delta = (geometry['head_offset'] - base_head) * 100.0
        depth = (geometry['shoulder_width'] / self.baseline['shoulder_width'] - 1.0) * 100.0
        self.embedding = np.array([delta[0], delta[1], depth], dtype=float)
        if np.isfinite(self.embedding).all():
            self.history.append(self.embedding.copy())

    def update(self, frame, min_submit_interval=0.0):
        updated = self._consume_result()
        if self.result_time is not None and time.monotonic()-self.result_time > 1.0:
            self.points = self.scores = None
            self.metrics = self.embedding = None
        now = time.monotonic()
        if self.future is None and now-self.last_submit_time >= min_submit_interval:
            self.submitted_time = time.monotonic()
            self.last_submit_time = self.submitted_time
            self.submitted_frame = frame.copy()
            self.future = self.executor.submit(self._infer, self.submitted_frame)
        if updated:
            self._update_values()
        # Keep the camera live; avoid painting stale keypoints on a moving user.
        camera = frame.copy()
        if self.points is not None and self.result_time is not None and now-self.result_time <= .15:
            camera = draw_skeleton(camera, self.points[None], self.scores[None],
                                   kpt_thr=self.threshold, openpose_skeleton=False,
                                   radius=4, line_width=3)
            camera = draw_camera(camera, measure(self.points, self.scores, self.baseline, self.threshold))
        return self._compose(camera)

    def set_baseline(self):
        if self.points is None or self.result_time is None or time.monotonic()-self.result_time > 1:
            print('[WARN] RTMPose baseline not set: no person detected.')
            return False
        pose = normalized_pose(self.points, self.scores, self.threshold)
        geometry = visible_geometry(self.points, self.scores, self.threshold)
        if pose[0] is None or geometry is None or geometry['head_offset'] is None:
            print('[WARN] RTMPose baseline not set: keep face and both shoulders visible.')
            return False
        self.baseline = {
            'pose': (pose[0].copy(), pose[1].copy()),
            'shoulder_width': geometry['shoulder_width'],
            'head_offset': geometry['head_offset'].copy(),
            'face_pitch_proxy': geometry['face_pitch_proxy'],
        }
        self.history.clear()
        self.embedding = np.zeros(3, dtype=float)
        self._update_values()
        print('[INFO] RTMPose personal posture baseline set.')
        return True

    def pose_vector(self):
        """Return a compact camera-relative pose vector for gaze correction.

        The values are observable 2D geometry and depth proxies, not anatomical
        3D joint angles. None is returned for stale or incomplete detections.
        """
        if (self.points is None or self.scores is None or self.result_frame is None or
                self.result_time is None or time.monotonic()-self.result_time > 1.0):
            return None
        geometry=visible_geometry(self.points,self.scores,self.threshold)
        if geometry is None or geometry['head_offset'] is None:
            return None
        h,w=self.result_frame.shape[:2]
        shoulder=geometry['shoulder']
        tilt=posture_metrics(self.points,self.scores,None,self.threshold)['shoulder_tilt_deg']
        pitch=geometry['face_pitch_proxy']
        if tilt is None or pitch is None:
            return None
        vector=np.array([shoulder[0]/w,shoulder[1]/h,
                         np.log(max(geometry['shoulder_width'],1.)/np.hypot(w,h)),
                         geometry['head_offset'][0],geometry['head_offset'][1],
                         pitch/45.,tilt/45.],dtype=float)
        return vector if np.isfinite(vector).all() else None

    def clear_baseline(self):
        self.baseline=None
        self.embedding=None
        self.metrics=None
        self.history.clear()

    def log(self, gaze_x=None, gaze_y=None):
        now = time.monotonic()
        if now - self.last_log < .1:
            return
        row = {name: None for name in self.fields}
        row.update(timestamp=time.time(), gaze_x_px=gaze_x, gaze_y_px=gaze_y)
        if self.embedding is not None:
            row.update(embedding_lateral_percent_sw=float(self.embedding[0]),
                       embedding_vertical_percent_sw=float(self.embedding[1]),
                       embedding_depth_proxy_percent=float(self.embedding[2]))
        if self.metrics:
            row.update(self.metrics)
        self.writer.writerow(row)
        self.handle.flush()
        self.last_log = now

    def _compose(self, camera):
        target_h = 540
        target_w = max(1, int(camera.shape[1] * target_h / camera.shape[0]))
        view = cv2.resize(camera, (target_w, target_h), interpolation=cv2.INTER_AREA)
        panel = np.full((target_h, 560, 3), 245, np.uint8)
        cv2.putText(panel, 'RTMPose + Gaze', (18, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    .72, (35, 70, 115), 2, cv2.LINE_AA)
        panel = draw_panel(panel, measure(self.points, self.scores, self.baseline, self.threshold))
        values = self.metrics or {}
        specs = (
            ('1. 머리 숙임 변화', 'head_bow_change_proxy_deg', 'deg'),
            ('2. 상체 전방 이동', 'upper_body_forward_change_percent', '%'),
            ('3. 어깨 기울기', 'shoulder_tilt_deg', 'deg'),
            ('4. 머리-어깨 상대 변화', 'head_shoulder_relative_change_percent', '%SW'),
            ('5. 개인 내 자세 변화량', 'personal_change_percent_shoulder_width', '%SW'))
        lines=[]
        for index,(label,key,unit) in enumerate(specs):
            value=values.get(key)
            shown='N/A' if value is None else f'{value:+.1f} {unit}'
            lines.append((label,(18,285+index*39),(45,45,45),False))
            lines.append((shown,(370,285+index*39),(145,35,80),False))
        status=('B: 개인 기준 자세 재설정' if self.baseline is not None else
                'B: 편안한 개인 기준 자세 설정')
        lines.extend([(status,(18,475),(0,105,30),False),
                      ('3D 좌표와 척추각이 아닌 영상 대리지표',(18,505),(185,25,25),False)])
        panel=put_korean(panel,lines)
        return np.hstack((view, panel))

    def close(self):
        if self.future is not None:
            try:
                self.future.result(timeout=5)
            except Exception:
                pass
        self.executor.shutdown(wait=True)
        self.handle.close()
        print('[INFO] Saved synchronized gaze/posture data:',
              self.session / 'gaze_posture_embedding.csv')
