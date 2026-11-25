import unittest
import numpy as np
from posture_protractor import measure,draw_camera,draw_panel

class ProtractorTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[320,190],[300,170],[340,170],[270,180],[370,180],[220,330],[420,330]],float)
        self.s=np.ones(7)

    def test_upright_and_shift(self):
        self.assertAlmostEqual(measure(self.p,self.s)['lateral'],0)
        self.p[:5,0]+=50
        self.assertGreater(measure(self.p,self.s)['lateral'],0)

    def test_rotation_relative_to_shoulders(self):
        angle=np.radians(20)
        r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        p=(self.p-[320,330])@r.T+[320,330]
        m=measure(p,self.s)
        self.assertAlmostEqual(m['lateral'],0,places=6)
        self.assertAlmostEqual(m['shoulder'],20,places=6)

    def test_reference_and_invalid(self):
        ref={'face_pitch_proxy':np.degrees(np.arctan2(20,40))}
        self.assertAlmostEqual(measure(self.p,self.s,ref)['pitch_change'],0)
        self.p[0,1]+=10
        self.assertGreater(measure(self.p,self.s,ref)['pitch_change'],0)
        self.s[5]=0
        self.assertIsNone(measure(self.p,self.s))
        self.assertIsNone(measure(None,None))

    def test_render(self):
        m=measure(self.p,self.s)
        a=np.zeros((480,640,3),np.uint8)
        self.assertGreater(draw_camera(a,m).sum(),0)
        self.assertEqual(draw_panel(np.zeros((540,560,3),np.uint8),None).shape,(540,560,3))

if __name__=='__main__': unittest.main()
