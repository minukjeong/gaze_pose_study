import unittest
import numpy as np
from pose_stabilizer import PoseStabilizer
from rtmpose_posture_live import personal_change


class PoseTests(unittest.TestCase):
    def test_missing_shoulders_after_baseline(self):
        baseline = (np.zeros((7,2)), np.ones(7,bool))
        self.assertIsNone(personal_change((None,None),baseline))

    def test_jitter_dropout_and_recovery(self):
        f = PoseStabilizer()
        p = np.tile([150.,100.],(17,1)); p[5]=[100,180]; p[6]=[220,180]
        s = np.ones(17)
        f.update(p,s,0,(480,640,3))
        noisy = p.copy(); noisy[0,0] += 5
        a,_ = f.update(noisy,s,.05,(480,640,3))
        self.assertLess(a[0,0]-p[0,0],5)
        noisy[0,0] += 100
        _,score = f.update(noisy,s,.1,(480,640,3))
        self.assertEqual(score[0],0)
        _,score = f.update(noisy,s,.2,(480,640,3))
        self.assertGreater(score[0],0)
        s[5]=0
        _,score = f.update(noisy,s,.3,(480,640,3))
        self.assertEqual(score[5],0)


if __name__ == '__main__': unittest.main()
