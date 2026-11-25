import unittest
from types import SimpleNamespace

import numpy as np

from dynamic_gaze import (ConditionalResidual, DynamicGazeCalibration, POSTURES,
                          TARGETS, NinePointCandidateCalibration, NinePointDynamicCalibration, head_features, pose_inputs)
from gaze_mapping import ScreenMap


class BaseMap:
    mean = np.array([.5, .5])
    scale = np.array([.3, .3])

    def predict(self, raw):
        return np.asarray(raw)


def trials(seed, bias=True):
    rng = np.random.default_rng(seed)
    result = []
    for index, (posture, _) in enumerate(POSTURES):
        for target in TARGETS:
            samples = []
            for _ in range(20):
                p = np.zeros(9)
                p[0] = (index-3)*.06+rng.normal(0, .001)
                p[1] = ((index % 3)-1)*.07+rng.normal(0, .001)
                delta = np.array([p[0]*(.4+target[0]*.2), p[1]*.4]) if bias else np.zeros(2)
                raw = np.array(target)+delta+rng.normal(0, .001, 2)
                samples.append(dict(raw=raw.tolist(), head=p.tolist(),
                                    head_shoulders=None, target=list(target)))
            result.append(dict(posture=posture, samples=samples, target=list(target)))
    return result


class DynamicGazeTests(unittest.TestCase):
    def test_nine_point_candidate_keeps_correction_outside_posture_support(self):
        calibration = NinePointCandidateCalibration(BaseMap(), [1920, 1080])
        calibration.models['head'] = ConditionalResidual().fit(BaseMap(), trials(1), 'head')
        for axis in (0, 1, 2, 4, 8):
            for direction in (-1, 1):
                pose = calibration.models['head'].center.copy()
                pose[axis] += direction * 2
                delta, status = calibration.predict([.5, .5],
                    {'head': pose, 'head_shoulders': None})
                self.assertEqual(status, 'nine_point_candidate:head:applied')
                self.assertTrue(np.isfinite(delta).all())
                self.assertTrue((abs(delta) <= .15).all())
        delta, status = calibration.predict([.5, .5],
            {'head': np.full(9, np.nan), 'head_shoulders': None})
        self.assertIn('base_fallback', status)
        np.testing.assert_array_equal(delta, [0, 0])

    def test_nine_points_complete_without_additional_measurements(self):
        targets = [(x, y) for y in (.12, .5, .88) for x in (.12, .5, .88)]
        base = ScreenMap().fit(targets, targets)
        data = [dict(target=list(t), samples=[dict(raw=list(t), target=list(t),
                    head=None, head_shoulders=None) for _ in range(12)]) for t in targets]
        calibration = NinePointDynamicCalibration(base, [1920, 1080]).fit(data)
        self.assertTrue(calibration.finished)
        self.assertEqual(len(calibration.trials['initial_nine']), 9)
        self.assertIsNone(calibration.selected)
        self.assertEqual(len(calibration.models['head'].report['folds']), 9)
        np.testing.assert_array_equal(calibration.predict([.5, .5],
            {'head': None, 'head_shoulders': None})[0], [0, 0])

    def test_same_frame_head_translation_rotation(self):
        face = SimpleNamespace(center=np.array([.1, .2, 1.]),
                               head_pose_rot=SimpleNamespace(as_matrix=lambda: np.eye(3)))
        value = head_features(face)
        self.assertEqual(value.shape, (9,))
        np.testing.assert_allclose(value[:3], [.1, .2, 0])
        face.center[2] = 0
        self.assertIsNone(head_features(face))

    def test_old_shoulders_are_excluded_without_losing_head(self):
        for age in (None, -.1, .151, 2.):
            value = pose_inputs(np.zeros(9), np.ones(7), age)
            self.assertIsNotNone(value['head'])
            self.assertIsNone(value['head_shoulders'])
        self.assertEqual(len(pose_inputs(np.zeros(9), np.ones(7), .1)['head_shoulders']), 16)

    def test_held_out_improvement_and_no_validation_fit(self):
        base = BaseMap()
        model = ConditionalResidual(alpha=.1).fit(base, trials(1), 'head')
        self.assertFalse(model.enabled)
        weights = model.weights.copy()
        model.validate(base, trials(2), 'head', np.array([1920, 1080]))
        self.assertTrue(model.enabled, model.report)
        self.assertGreater(model.report['improvement_percent'], 50)
        np.testing.assert_array_equal(model.weights, weights)
        sample = trials(3)[0]['samples'][0]
        delta, reason = model.correction(sample['raw'], sample['head'])
        self.assertEqual(reason, 'applied')
        self.assertGreater(np.linalg.norm(delta), 0)

    def test_worsening_validation_disables_correction(self):
        model = ConditionalResidual(alpha=.1).fit(BaseMap(), trials(1), 'head')
        model.validate(BaseMap(), trials(4, bias=False), 'head', np.array([1920, 1080]))
        self.assertFalse(model.enabled)
        np.testing.assert_array_equal(model.correction([.5, .5], np.zeros(9))[0], [0, 0])

    def test_outside_range_and_nonfinite_fall_back(self):
        model = ConditionalResidual().fit(BaseMap(), trials(1), 'head')
        model.enabled = True
        for p in (None, np.full(9, 100), np.full(9, np.nan), np.zeros(7)):
            value, reason = model.correction([.5, .5], p)
            np.testing.assert_array_equal(value, [0, 0])
            self.assertNotEqual(reason, 'applied')

    def test_head_fallback_for_missing_shoulders(self):
        calibration = DynamicGazeCalibration(BaseMap(), [1920, 1080])
        calibration.models['head'] = ConditionalResidual().fit(BaseMap(), trials(1), 'head')
        calibration.models['head'].enabled = True
        s = trials(3)[0]['samples'][0]
        _, reason = calibration.predict(s['raw'], {'head': s['head'], 'head_shoulders': None})
        self.assertEqual(reason, 'head')

    def test_protocol_finishes_with_disjoint_validation(self):
        calibration = DynamicGazeCalibration(BaseMap(), [1920, 1080])
        now = 0.
        for phase in ('adapt', 'validate'):
            for i in range(len(POSTURES)*len(TARGETS)):
                self.assertEqual(calibration.phase, phase)
                calibration.confirm()
                pose = np.zeros(9)
                pose[0] = (i//5-3)*.06
                target = np.array((TARGETS if phase == 'adapt' else TARGETS[::-1])[i % 5])
                for step in range(42):
                    calibration.update(now+step*.1, target,
                                       {'head': pose.tolist(), 'head_shoulders': None})
                    if calibration.finished or calibration.index != i or calibration.phase != phase:
                        break
                now += 5
        self.assertTrue(calibration.finished)
        self.assertEqual(len(calibration.trials['adapt']), len(POSTURES)*len(TARGETS))
        self.assertEqual(len(calibration.trials['validate']), len(POSTURES)*len(TARGETS))
        self.assertFalse(calibration.routing_report['enabled'])


if __name__ == '__main__':
    unittest.main()
