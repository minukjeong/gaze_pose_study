import unittest
import numpy as np
from gaze_visualization import build_gaze_visualization


class GazeVisualizationTests(unittest.TestCase):
    def test_path_and_dwell_summary(self):
        records=[]
        for i in range(20): records.append((i*.1,100+i%2,100))
        for i in range(10): records.append((2+i*.1,100+i*30,100+i*10))
        for i in range(20): records.append((3+i*.1,450+i%2,220))
        image,summary=build_gaze_visualization(records,640,360)
        self.assertEqual(image.shape,(360,640,3))
        self.assertEqual(summary['samples'],50)
        self.assertGreaterEqual(len(summary['dwells']),2)
        self.assertTrue(np.isfinite(image).all())


if __name__ == '__main__': unittest.main()
