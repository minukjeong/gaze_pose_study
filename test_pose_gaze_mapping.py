import unittest

import numpy as np

from gaze_mapping import PoseCorrectionModel, ScreenMap


class PoseGazeMappingTests(unittest.TestCase):
    def test_pose_residual_is_enabled_only_after_held_out_improvement(self):
        rng=np.random.default_rng(7)
        targets=np.array([(x,y) for y in (.12,.5,.88) for x in (.12,.5,.88)])
        trials=[]; centers=[]
        for target in targets:
            raw=[]; poses=[]
            for _ in range(24):
                pose=np.zeros(7);pose[0]=rng.uniform(-.08,.08)
                gaze=(target-.5)/2 + np.array([pose[0]*.7,pose[0]*-.35])
                raw.append(gaze.tolist());poses.append(pose.tolist())
            trials.append({'target':target.tolist(),'raw':raw,'pose':poses})
            centers.append(np.median(raw,axis=0))
        base=ScreenMap().fit(centers,targets)
        correction=PoseCorrectionModel(alpha=1.).fit(base,trials)
        self.assertTrue(correction.enabled,correction.report)
        self.assertGreater(correction.report['improvement_percent'],2.)

    def test_missing_pose_falls_back_to_zero(self):
        correction=PoseCorrectionModel()
        np.testing.assert_allclose(correction.predict(None),[0,0])


if __name__=='__main__':
    unittest.main()
