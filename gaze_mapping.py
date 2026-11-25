"""Direct screen mapping; calibration and validation remain separate."""
import numpy as np


def design(x):
    x = np.atleast_2d(x)
    a, b = x[:, 0], x[:, 1]
    return np.c_[np.ones(len(x)), a, b, a*a, a*b, b*b]


class ScreenMap:
    def __init__(self, min_samples=9):
        self.min_samples = min_samples

    def fit(self, features, targets):
        x, y = np.asarray(features), np.asarray(targets)
        if len(x) < self.min_samples or not np.isfinite(x).all():
            raise ValueError(f'Need at least {self.min_samples} valid fixation targets')
        self.mean = x.mean(0)
        self.scale = x.std(0)
        if np.any(self.scale < 1e-5):
            raise ValueError('Insufficient gaze variation; repeat calibration')
        d = design((x-self.mean)/self.scale)
        if np.linalg.matrix_rank(d) < 6:
            raise ValueError('Degenerate gaze calibration')
        penalty = np.eye(6)*.05
        penalty[0, 0] = 0
        self.weights = np.linalg.solve(d.T@d+penalty, d.T@y)
        return self

    def predict(self, x):
        return (design((np.asarray(x)-self.mean)/self.scale)@self.weights)[0]

    def as_dict(self):
        return {k: getattr(self, k).tolist() for k in ('mean', 'scale', 'weights')}


class PoseCorrectionModel:
    """Regularized residual correction learned from pose variation in 9 targets."""
    def __init__(self, alpha=8.0, minimum_improvement=.02):
        self.alpha = float(alpha)
        self.minimum_improvement = float(minimum_improvement)
        self.enabled = False
        self.center = self.scale = self.weights = None
        self.report = {}

    @staticmethod
    def _paired(trials):
        pose=[]; residual_target=[]; raw=[]
        for trial in trials:
            target=np.asarray(trial['target'],float)
            for gaze, posture in zip(trial['raw'],trial['pose']):
                if posture is None:
                    continue
                p=np.asarray(posture,float); g=np.asarray(gaze,float)
                if np.isfinite(p).all() and np.isfinite(g).all():
                    pose.append(p); raw.append(g); residual_target.append(target)
        return np.asarray(raw),np.asarray(pose),np.asarray(residual_target)

    def _fit_residual(self, base, trials):
        raw,pose,target=self._paired(trials)
        if len(pose)<18:
            raise ValueError('Need at least 18 gaze samples with valid pose')
        center=np.median(pose,axis=0)
        mad=1.4826*np.median(np.abs(pose-center),axis=0)
        floors=np.array([.01,.01,.02,.02,.02,.03,.03])
        scale=np.maximum(mad,floors)
        design=np.clip((pose-center)/scale,-3.,3.)
        residual=np.asarray([t-base.predict(g) for g,t in zip(raw,target)])
        penalty=np.eye(design.shape[1])*self.alpha
        weights=np.linalg.solve(design.T@design+penalty,design.T@residual)
        return center,scale,weights

    @staticmethod
    def _correction(pose,center,scale,weights):
        z=np.clip((np.asarray(pose,float)-center)/scale,-3.,3.)
        return np.clip(z@weights,-.15,.15)

    def fit(self, base, trials):
        if len(trials)!=9:
            raise ValueError('Pose correction requires exactly nine target trials')
        base_errors=[]; fused_errors=[]; folds=[]
        try:
            for held_out in range(len(trials)):
                train=[trial for i,trial in enumerate(trials) if i!=held_out]
                centers=[np.median(np.asarray(t['raw'],float),axis=0) for t in train]
                targets=[t['target'] for t in train]
                fold_base=ScreenMap(min_samples=6).fit(centers,targets)
                center,scale,weights=self._fit_residual(fold_base,train)
                trial=trials[held_out]; target=np.asarray(trial['target'],float)
                fold_base_error=[]; fold_fused_error=[]
                for gaze,pose in zip(trial['raw'],trial['pose']):
                    if pose is None:
                        continue
                    first=fold_base.predict(gaze)
                    second=first+self._correction(pose,center,scale,weights)
                    fold_base_error.append(float(np.linalg.norm(first-target)))
                    fold_fused_error.append(float(np.linalg.norm(second-target)))
                if fold_base_error:
                    base_errors.extend(fold_base_error); fused_errors.extend(fold_fused_error)
                    folds.append({'target':trial['target'],
                                  'base_mean_normalized':float(np.mean(fold_base_error)),
                                  'fused_mean_normalized':float(np.mean(fold_fused_error)),
                                  'samples':len(fold_base_error)})
            if not base_errors:
                raise ValueError('No held-out samples with valid pose')
            base_mean=float(np.mean(base_errors)); fused_mean=float(np.mean(fused_errors))
            improvement=(base_mean-fused_mean)/max(base_mean,1e-9)
            self.enabled=bool(improvement>=self.minimum_improvement)
            self.center,self.scale,self.weights=self._fit_residual(base,trials)
            self.report={'method':'leave-one-target-out internal cross-validation',
                         'base_mean_normalized':base_mean,
                         'fused_mean_normalized':fused_mean,
                         'improvement_percent':float(improvement*100),
                         'enabled':self.enabled,'folds':folds,
                         'warning':'Internal validation on the same nine-target session; not an independent accuracy test.'}
        except (ValueError,np.linalg.LinAlgError) as error:
            self.enabled=False
            self.report={'method':'leave-one-target-out internal cross-validation',
                         'enabled':False,'reason':str(error),
                         'warning':'Pose correction unavailable; base gaze mapping is used.'}
        return self

    def predict(self, pose):
        if not self.enabled or pose is None:
            return np.zeros(2,float)
        return self._correction(pose,self.center,self.scale,self.weights)

    def as_dict(self):
        return {'alpha':self.alpha,'minimum_improvement':self.minimum_improvement,
                'enabled':self.enabled,'center':None if self.center is None else self.center.tolist(),
                'scale':None if self.scale is None else self.scale.tolist(),
                'weights':None if self.weights is None else self.weights.tolist(),
                'report':self.report}


def fixation_quality(samples):
    """Return robust fixation and diagnostics in gaze-slope coordinates."""
    info = {'samples': len(samples), 'retained': 0, 'spread': None, 'reason': 'too_few_samples'}
    if len(samples) < 6:
        return None, info
    a = np.asarray(samples)
    median = np.median(a, axis=0)
    mad = np.maximum(np.median(abs(a-median), axis=0), .003)
    kept = a[(abs(a-median) <= 3.5*mad).all(axis=1)]
    info['retained'] = len(kept)
    if len(kept) < 6 or len(kept) < .65*len(a):
        info['reason'] = 'too_many_outliers'
        return None, info
    center = np.median(kept, axis=0)
    radius = np.linalg.norm(kept-center, axis=1)
    info['spread'] = float(np.quantile(radius, .90))
    if info['spread'] > .18:
        info['reason'] = 'unstable_fixation'
        return None, info
    info['reason'] = 'accepted'
    return center, info


def fixation(samples):
    return fixation_quality(samples)[0]
