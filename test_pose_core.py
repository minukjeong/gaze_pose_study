"""Synthetic correctness tests, never evidence of human gaze accuracy."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pose_core import PoseGazeModel, pose_features, evaluate_rows, aggregate_trial


class CoreTests(unittest.TestCase):
    def samples(self):
        rng=np.random.default_rng(71)
        trials=[]
        for i in range(144):
            p=rng.normal(size=6);raw=rng.uniform(-1,1,size=2)
            target=.5+.2*raw+np.array([.04*p[0],.04*p[1]])
            trials.append(dict(phase='train',target=target,raw=raw,pose=p,posture='A' if i%2 else 'B'))
        return trials

    def test_roundtrip_and_missing_pose(self):
        trials=self.samples();model=PoseGazeModel.fit(trials,{'test':'synthetic'})
        expected=model.predict(trials[0]['raw'],trials[0]['pose'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.json';model.save(path);loaded=PoseGazeModel.load(path)
            np.testing.assert_allclose(expected['fused'],loaded.predict(trials[0]['raw'],trials[0]['pose'])['fused'])
        missing=model.predict(trials[0]['raw'])
        np.testing.assert_allclose(missing['baseline'],missing['fused'])
        self.assertFalse(missing['fusion_used'])

    def test_no_evaluation_fit(self):
        trials=self.samples();trials[0]['phase']='evaluate'
        with self.assertRaises(ValueError):PoseGazeModel.fit(trials,{})

    def test_independent_synthetic_pose_bias(self):
        model=PoseGazeModel.fit(self.samples(),{})
        rng=np.random.default_rng(923);base=[];fused=[]
        for _ in range(100):
            p=rng.normal(size=6);raw=rng.uniform(-1,1,size=2)
            target=.5+.2*raw+np.array([.04*p[0],.04*p[1]])
            pred=model.predict(raw,p)
            base.append(np.linalg.norm(pred['baseline']-target));fused.append(np.linalg.norm(pred['fused']-target))
        self.assertLess(np.mean(fused),np.mean(base)*.5)

    def test_shoulder_gate_and_translation_scale(self):
        rng=np.random.default_rng(8);world=rng.normal(size=(33,3))
        image=np.full((33,4),.5);image[:,3]=1;image[11,0]=.3;image[12,0]=.7
        a=pose_features(world,image,np.eye(3),np.zeros(3))
        b=pose_features(world*3+17,image,np.eye(3),np.zeros(3))
        np.testing.assert_allclose(a,b,atol=1e-12)
        image[11,3]=.1
        self.assertIsNone(pose_features(world,image,np.eye(3),np.zeros(3)))

    def test_trial_and_unclipped_error(self):
        rows=[dict(phase='evaluate',posture='A',trial_id='1',target=[0,0],
                   baseline=[2,0],fused=[1,0],fusion_used=True) for _ in range(10)]
        out=evaluate_rows(rows,100,100)['overall']
        self.assertEqual(out['baseline_mean_px'],200)
        self.assertEqual(out['fused_mean_px'],100)
        self.assertIsNone(aggregate_trial([{'raw':None}]*10,'train','1','A','unknown',[.5,.5]))


if __name__=='__main__':unittest.main()
