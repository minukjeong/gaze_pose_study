"""Confidence gating and time-based filtering, once per inference result."""
import numpy as np


class PoseStabilizer:
    def __init__(self, threshold=.4):
        self.threshold = threshold
        self.points = None
        self.valid = None
        self.time = None
        self.pending = None

    def update(self, points, scores, timestamp, shape):
        p = np.asarray(points, dtype=float).copy()
        s = np.asarray(scores, dtype=float).copy()
        h,w = shape[:2]
        valid = (np.isfinite(p).all(1) & np.isfinite(s) & (s >= self.threshold)
                 & (p[:,0]>=0) & (p[:,0]<w) & (p[:,1]>=0) & (p[:,1]<h))
        if self.points is not None and timestamp-self.time < 1:
            common = valid & self.valid
            scale = max(40., np.linalg.norm(self.points[5]-self.points[6]))
            jump = common & (np.linalg.norm(p-self.points,axis=1) > .35*scale)
            # A large movement must persist into another observation; do not
            # freeze true sustained motion indefinitely.
            confirmed = np.zeros(len(p),bool) if self.pending is None else np.linalg.norm(p-self.pending,axis=1)<.15*scale
            valid &= ~(jump & ~confirmed)
            alpha = 1-np.exp(-max(.001,timestamp-self.time)/.10)
            smooth = common & valid & ~jump
            self.pending = p.copy()
            p[smooth] = self.points[smooth]+alpha*(p[smooth]-self.points[smooth])
        else:
            self.pending = p.copy()
        p[~np.isfinite(p)] = 0
        s[~valid] = 0
        self.points,self.valid,self.time = p.copy(),valid,timestamp
        return p,s
