import unittest
import numpy as np
from gaze_mapping import ScreenMap, fixation, fixation_quality


class MappingTests(unittest.TestCase):
    def test_unseen_targets(self):
        x = np.array([(a,b) for a in (-.2,0,.2) for b in (-.2,0,.2)])
        y = x*2+.5
        m = ScreenMap().fit(x,y)
        for p in ((.1,-.1),(-.1,.1)):
            self.assertLess(np.linalg.norm(m.predict(p)-(np.array(p)*2+.5)),.005)

    def test_no_variation_rejected(self):
        with self.assertRaises(ValueError):
            ScreenMap().fit(np.zeros((9,2)),np.zeros((9,2)))

    def test_missing_and_unstable(self):
        self.assertIsNone(fixation([[0,0]]*5))
        self.assertIsNone(fixation([[.2,0],[-.2,0]]*10))

    def test_outliers(self):
        samples = [[.1,.2]]*18+[[1,1]]*2
        np.testing.assert_allclose(fixation(samples),[.1,.2])

    def test_diagnostics(self):
        value, info = fixation_quality([[.1,.2]]*6)
        self.assertEqual(info['reason'],'accepted')
        np.testing.assert_allclose(value,[.1,.2])


if __name__ == '__main__': unittest.main()
