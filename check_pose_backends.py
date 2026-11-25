from pathlib import Path
import cv2
import numpy as np
from rtmlib import Body
from hybrid_pose import HybridPose
import time


def main():
    root = Path(__file__).resolve().parent
    cache = Path.home()/'.cache/rtmlib/hub/checkpoints'
    frame = cv2.imdecode(np.fromfile(r'C:\Users\alsdn\OneDrive\Desktop\스크린샷 2026-09-09 200158.png',dtype=np.uint8),cv2.IMREAD_COLOR)
    frame = frame[:,:int(frame.shape[1]*.65)].copy()
    cpu = Body(mode='lightweight',backend='onnxruntime',device='cpu')
    npu = HybridPose()
    box = cpu.det_model(frame)
    if len(box)==0: box=np.array([[0,0,frame.shape[1],frame.shape[0]]])
    print('BOX CPU',box,'BOX NPU',npu.det_model(frame))
    for name, m in [('CPU',cpu),('NPU',npu)]:
        k,s=m.pose_model(frame,box)
        print(name,'HEAD',np.round(k[0,:7],1),'SCORES',np.round(s[0,:7],2))
    a,center,scale=cpu.pose_model.preprocess(frame,box[0])
    x=cpu.pose_model.inference(a)
    y=npu.pose_model.inference(a)
    for i in range(2):
        print('OUTPUT',i,'MAX_DIFF',np.max(abs(x[i]-y[i])), 'ARGMAX_CPU',x[i].argmax(-1)[0,:7], 'ARGMAX_NPU',y[i].argmax(-1)[0,:7])
    for _ in range(3):
        start=time.perf_counter(); k,s=npu(frame)
        print('HYBRID_MS',round((time.perf_counter()-start)*1000,1))
    print('HYBRID_HEAD',np.round(k[0,:7],1))


if __name__ == '__main__': main()
