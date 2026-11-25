"""NPU person detection with numerically verified CPU keypoint inference."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
from rtmlib import YOLOX, RTMPose


class HybridPose:
    def __init__(self):
        cache = Path.home()/'.cache/rtmlib/hub/checkpoints'
        self.det_model = YOLOX(str(cache/'yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx'),
                              model_input_size=(416,416),backend='openvino',device='npu')
        path = cache/'rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx'
        self.pose_model = RTMPose(str(path),model_input_size=(192,256),backend='onnxruntime',device='cpu')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.pose_model.session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def __call__(self, frame):
        boxes = np.asarray(self.det_model(frame))
        if not len(boxes):
            return np.empty((0,17,2)),np.empty((0,17))
        area = np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
        return self.pose_model(frame,boxes[[int(np.argmax(area))]])
