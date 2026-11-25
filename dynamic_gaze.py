"""Personal gaze/pose residuals with a separate, later validation pass.

Head center is model-derived, not measured depth. Shoulder inputs are 2D
proxies. No anatomical neck angles or unseen eye directions are inferred.
"""
import numpy as np

from gaze_mapping import ScreenMap, fixation_quality


POSTURES = (
    ('neutral', 'Sit normally'),
    ('forward', 'Move head / upper body slightly forward'),
    ('backward', 'Move head / upper body slightly backward'),
    ('left', 'Move head / upper body slightly left'),
    ('right', 'Move head / upper body slightly right'),
    ('bow', 'Lower your head slightly; keep eyes visible'),
    ('tilt', 'Tilt head / shoulders slightly to one side'),
)
TARGETS = ((.15, .15), (.85, .15), (.5, .5), (.15, .85), (.85, .85))


def head_features(face):
    """Same-frame camera-relative translation + continuous rotation features."""
    try:
        center = np.asarray(face.center, float).reshape(3)
        rotation = np.asarray(face.head_pose_rot.as_matrix(), float)
        if center[2] <= 0 or rotation.shape != (3, 3):
            return None
        value = np.r_[center[:2]/center[2], np.log(center[2]),
                      rotation[:, :2].ravel()]
        return value if np.isfinite(value).all() else None
    except (AttributeError, TypeError, ValueError):
        return None


def pose_inputs(head, shoulder, pose_age):
    """Never attach a shoulder result older than 150 ms to current gaze."""
    if head is None:
        return {'head': None, 'head_shoulders': None}
    combined = None
    if shoulder is not None and pose_age is not None and 0 <= pose_age <= .15:
        combined = np.r_[head, shoulder].tolist()
    return {'head': np.asarray(head).tolist(), 'head_shoulders': combined}


class ConditionalResidual:
    def __init__(self, alpha=5.):
        self.alpha = alpha
        self.enabled = False
        self.fitted = False
        self.report = {}

    def _design(self, raw, pose):
        z = np.clip((np.atleast_2d(pose)-self.center)/self.scale, -4, 4)
        g = np.clip((np.atleast_2d(raw)-self.gaze_center)/self.gaze_scale, -4, 4)
        return np.c_[z, z*g[:, :1], z*g[:, 1:2]]

    def fit(self, base, trials, key, minimum_trials=20, minimum_samples=8, require_variation=True):
        groups = [[s for s in t['samples'] if s[key] is not None] for t in trials]
        groups = [g for g in groups if len(g) >= minimum_samples]
        if len(groups) < minimum_trials:
            self.report = {'enabled': False, 'reason': 'insufficient fresh pose trials'}
            return self
        # Equal total weight per fixation, regardless of camera speed.
        samples = [s for g in groups for s in g]
        raw = np.array([s['raw'] for s in samples])
        pose = np.array([s[key] for s in samples])
        self.center = np.median(pose, axis=0)
        self.scale = np.maximum(1.4826*np.median(abs(pose-self.center), axis=0), .02)
        centers = np.array([np.median([s[key] for s in g], axis=0) for g in groups])
        if require_variation and np.max(np.ptp(centers, axis=0)) < .04:
            self.report = {'enabled': False, 'reason': 'insufficient observed posture variation'}
            return self
        self.gaze_center, self.gaze_scale = base.mean.copy(), base.scale.copy()
        target = np.array([s['target'] for s in samples])
        residual = target-np.array([base.predict(g) for g in raw])
        d = self._design(raw, pose)
        weight = np.concatenate([np.full(len(g), 1/len(g)) for g in groups])
        self.weights = np.linalg.solve(d.T@(weight[:, None]*d)+np.eye(d.shape[1])*self.alpha,
                                       d.T@(weight[:, None]*residual))
        normalized = (pose-self.center)/self.scale
        self.support = normalized[::max(1, len(normalized)//400)].copy()
        self.lower, self.upper = normalized.min(0)-.5, normalized.max(0)+.5
        self.fitted = True
        self.report = {'enabled': False, 'reason': 'awaiting separate validation',
                       'training_trials': len(groups), 'samples': len(samples)}
        return self

    def correction(self, raw, pose, candidate=False, extrapolate=False):
        zero = np.zeros(2)
        if not self.fitted or (not candidate and not self.enabled):
            return zero, 'not_validated'
        if pose is None or raw is None:
            return zero, 'pose_missing'
        pose, raw = np.asarray(pose, float), np.asarray(raw, float)
        if (pose.shape != self.center.shape or raw.shape != (2,) or
                not np.isfinite(pose).all() or not np.isfinite(raw).all()):
            return zero, 'invalid_input'
        z = (pose-self.center)/self.scale
        if not extrapolate and (np.any(z < self.lower) or np.any(z > self.upper)):
            return zero, 'outside_training_postures'
        distance = np.min(np.sqrt(np.mean((self.support-z)**2, axis=1)))
        if not extrapolate and distance >= 1.5:
            return zero, 'outside_training_postures'
        if not extrapolate and np.any(abs((raw-self.gaze_center)/self.gaze_scale) > 4):
            return zero, 'outside_training_gaze'
        # Fade toward the base map near the edge of observed support.
        confidence = 1. if extrapolate else min(1., max(0., (1.5-distance)/.5))
        delta = (self._design(raw, pose)@self.weights)[0]*confidence
        return np.clip(delta, -.15, .15), 'applied'

    def validate(self, base, trials, key, screen):
        rows = []
        used = total = 0
        for trial in trials:
            before, after = [], []
            for s in trial['samples']:
                first = base.predict(s['raw'])
                delta, reason = self.correction(s['raw'], s[key], candidate=True)
                target = np.asarray(s['target'])
                before.append(float(np.linalg.norm((first-target)*screen)))
                after.append(float(np.linalg.norm((first+delta-target)*screen)))
                total += 1
                used += reason == 'applied'
            if before:
                rows.append({'posture': trial['posture'], 'base_px': float(np.mean(before)),
                             'corrected_px': float(np.mean(after))})
        if not rows:
            self.report.update(enabled=False, reason='no validation samples')
            return
        b, c = np.mean([[r['base_px'], r['corrected_px']] for r in rows], axis=0)
        improvement = float((b-c)/max(b, 1e-9))
        by_posture = {}
        for posture, _ in POSTURES:
            group = [r for r in rows if r['posture'] == posture]
            if group:
                bp, cp = np.mean([[r['base_px'], r['corrected_px']] for r in group], axis=0)
                by_posture[posture] = {'base_px': float(bp), 'corrected_px': float(cp)}
        adequate = len(rows) == len(POSTURES)*len(TARGETS) and used/max(total, 1) >= .6
        stable = all(r['corrected_px'] <= r['base_px']*1.05+2 for r in by_posture.values())
        self.enabled = bool(self.fitted and adequate and stable and improvement >= .02)
        self.report.update(enabled=self.enabled, base_mean_px=float(b), corrected_mean_px=float(c),
                           improvement_percent=improvement*100, coverage=used/max(total, 1),
                           by_posture=by_posture, validation_trials=len(rows),
                           reason='validated' if self.enabled else 'validation gate did not pass',
                           method='later held-out fixation pass; equal weight per trial',
                           warning='Same user/session; instructed targets, not eye-tracker ground truth.')

    def as_dict(self):
        result = {'enabled': self.enabled, 'report': self.report, 'alpha': self.alpha}
        if self.fitted:
            for name in ('center', 'scale', 'gaze_center', 'gaze_scale', 'weights', 'support', 'lower', 'upper'):
                result[name] = getattr(self, name).tolist()
        return result


class DynamicGazeCalibration:
    """User-confirmed posture changes; disjoint training and validation trials."""
    def __init__(self, base, screen):
        self.base, self.screen = base, np.asarray(screen)
        self.models = {key: ConditionalResidual() for key in ('head', 'head_shoulders')}
        self.phase, self.index = 'adapt', 0
        self.waiting = True
        self.started = None
        self.samples = []
        self.trials = {'adapt': [], 'validate': []}
        self.last_sample = -1e9
        self.status = 'Press SPACE when ready'
        self.finished = False

    def confirm(self):
        if self.waiting:
            self.waiting = False
            self.started = None

    def update(self, now, raw, inputs):
        posture, instruction = POSTURES[self.index//len(TARGETS)]
        target_index = self.index % len(TARGETS)
        # Reverse order in the later validation pass.
        targets = TARGETS if self.phase == 'adapt' else TARGETS[::-1]
        target = np.array(targets[target_index])
        message = f'{self.phase.upper()} {self.index+1}/{len(POSTURES)*len(TARGETS)}: {instruction}'
        if self.waiting:
            return target, message, 'Look at target; hold a comfortable pose; SPACE to start'
        if self.started is None:
            self.started = now
        elapsed = now-self.started
        if elapsed >= 1.5 and now-self.last_sample >= .08 and raw is not None and inputs['head'] is not None:
            self.samples.append(dict(raw=np.asarray(raw).tolist(), target=target.tolist(), **inputs))
            self.last_sample = now
        self.status = f'Hold pose and look at red target: {elapsed:.1f}s | samples {len(self.samples)}/12'
        if (elapsed >= 4 and len(self.samples) >= 12) or elapsed >= 10:
            accepted, quality = fixation_quality([s['raw'] for s in self.samples])
            if len(self.samples) < 12:
                accepted = None
                quality['reason'] = 'need 12 valid face/gaze samples'
            if accepted is None:
                self.status = 'Retry: '+quality['reason']+'; SPACE when ready'
                self.waiting = True
            else:
                self.trials[self.phase].append({'posture': posture, 'target': target.tolist(),
                                               'samples': self.samples})
                self.index += 1
                self.waiting = self.index % len(TARGETS) == 0
                if self.index == len(POSTURES)*len(TARGETS):
                    if self.phase == 'adapt':
                        for key, model in self.models.items():
                            model.fit(self.base, self.trials['adapt'], key)
                        self.phase, self.index = 'validate', 0
                    else:
                        for key, model in self.models.items():
                            model.validate(self.base, self.trials['validate'], key, self.screen)
                        self._validate_routing()
                        self.finished = True
            self.samples = []
            self.started = None
        return target, message, self.status

    def predict(self, raw, inputs):
        # Prefer fresh shoulders; missing/unsupported shoulders fall back to head.
        for key in ('head_shoulders', 'head'):
            delta, reason = self.models[key].correction(raw, inputs[key])
            if reason == 'applied':
                return delta, key
        return np.zeros(2), 'base: pose missing, unsupported or validation off'

    def _validate_routing(self):
        """Check actual head/shoulder fallback policy as well as each model."""
        rows = []
        for trial in self.trials['validate']:
            errors = []
            for sample in trial['samples']:
                base = self.base.predict(sample['raw'])
                delta, _ = self.predict(sample['raw'], sample)
                target = np.asarray(sample['target'])
                errors.append([np.linalg.norm((base-target)*self.screen),
                               np.linalg.norm((base+delta-target)*self.screen)])
            rows.append((trial['posture'], np.mean(errors, axis=0)))
        before, after = np.mean([values for _, values in rows], axis=0)
        stable = True
        for posture, _ in POSTURES:
            b, c = np.mean([values for p, values in rows if p == posture], axis=0)
            stable &= c <= b*1.05+2
        passed = bool(any(model.enabled for model in self.models.values())
                      and stable and after <= before*.98)
        if not passed:
            for model in self.models.values():
                model.enabled = False
                model.report['enabled'] = False
                model.report['routing_gate'] = 'combined fallback policy did not improve enough'
        self.routing_report = {'enabled': passed, 'base_mean_px': float(before),
                               'candidate_mean_px': float(after),
                               'improvement_percent': float((before-after)/max(before, 1e-9)*100)}

    def as_dict(self):
        return {'schema': 'dynamic-gaze-v1', 'models': {k: m.as_dict() for k, m in self.models.items()},
                'routing_validation': getattr(self, 'routing_report', None),
                'max_shoulder_age_s': .15, 'correction_limit_normalized': .15}


class NinePointDynamicCalibration(DynamicGazeCalibration):
    """Reuse the initial nine fixations; no extra user-facing targets."""
    def fit(self, trials):
        if len(trials) != 9:
            raise ValueError('Exactly nine accepted fixations are required')
        records = {key: [] for key in self.models}
        for held in range(9):
            train = [t for i, t in enumerate(trials) if i != held]
            try:
                centers = [fixation_quality([s['raw'] for s in t['samples']])[0] for t in train]
                base = ScreenMap(min_samples=6).fit(centers, [t['target'] for t in train])
                candidates = {key: ConditionalResidual().fit(base, train, key, minimum_trials=6)
                              for key in self.models}
            except (ValueError, np.linalg.LinAlgError):
                continue
            for key, candidate in candidates.items():
                errors = []
                used = 0
                for s in trials[held]['samples']:
                    first = base.predict(s['raw'])
                    delta, reason = candidate.correction(s['raw'], s[key], candidate=True)
                    target = np.asarray(s['target'])
                    errors.append([np.linalg.norm((first-target)*self.screen),
                                   np.linalg.norm((first+delta-target)*self.screen)])
                    used += reason == 'applied'
                b, c = np.mean(errors, axis=0)
                records[key].append(dict(base_px=float(b), corrected_px=float(c),
                                         coverage=used/max(1, len(errors))))
        for key in self.models:
            model = ConditionalResidual().fit(self.base, trials, key, minimum_trials=6)
            rows = records[key]
            enabled = False
            report = dict(method='leave-one-target-out internal cross-validation',
                          warning='Same nine fixations; not independent validation of new postures.',
                          folds=rows, enabled=False)
            if len(rows) == 9:
                b = float(np.mean([r['base_px'] for r in rows]))
                c = float(np.mean([r['corrected_px'] for r in rows]))
                coverage = float(np.mean([r['coverage'] for r in rows]))
                enabled = bool(model.fitted and b > 1e-9 and c <= .98*b and coverage >= .6
                               and all(r['corrected_px'] <= r['base_px']*1.05+2 for r in rows))
                report.update(base_mean_px=b, corrected_mean_px=c, coverage=coverage,
                              improvement_percent=(b-c)/max(b, 1e-9)*100)
            model.enabled = enabled
            report.update(enabled=enabled, reason='internal CV passed' if enabled else
                          model.report.get('reason', 'internal CV did not pass'))
            if model.fitted and not enabled:
                report['reason'] = 'insufficient internal CV improvement or coverage'
            model.report = report
            self.models[key] = model
        # Select one validated route; do not claim validation of an untested mix.
        eligible = [key for key, model in self.models.items() if model.enabled]
        self.selected = min(eligible, key=lambda k: self.models[k].report['corrected_mean_px']) if eligible else None
        self.routing_report = dict(selected=self.selected, enabled=self.selected is not None,
                                   method='selection on internal nine-target CV; no independent test')
        self.finished = True
        self.trials = {'initial_nine': trials}
        return self

    def predict(self, raw, inputs):
        if self.selected is not None:
            delta, reason = self.models[self.selected].correction(raw, inputs[self.selected])
            if reason == 'applied':
                return delta, self.selected
            return np.zeros(2), 'base: '+reason
        return np.zeros(2), 'base: internal CV did not enable correction'

    def as_dict(self):
        result = super().as_dict()
        result['schema'] = 'nine-point-dynamic-gaze-v1'
        return result


class NinePointCandidateCalibration(NinePointDynamicCalibration):
    """Reuse observed nine-point pose data without an improvement gate."""
    def fit(self, trials):
        super().fit(trials)
        self.internal_validation = self.routing_report.copy()
        for key in self.models:
            report = self.models[key].report.copy()
            candidate = ConditionalResidual().fit(
                self.base, trials, key, minimum_trials=1,
                minimum_samples=1, require_variation=False)
            training = candidate.report.copy()
            candidate.report = dict(report, training=training,
                candidate_applied_when_available=candidate.fitted,
                activation_policy='observed nine-point data; improvement gate bypassed')
            self.models[key] = candidate
        self.selected = next((k for k in ('head_shoulders','head') if self.models[k].fitted), None)
        self.routing_report = dict(enabled=self.selected is not None, selected=self.selected,
            policy='nine-point candidate; not independently validated',
            internal_validation=self.internal_validation)
        return self

    def predict(self, raw, inputs):
        reasons=[]
        for key in ('head_shoulders','head'):
            # Keep the nine-point calibration active as posture changes. The
            # design clips normalized features and correction remains bounded.
            delta,reason=self.models[key].correction(
                raw,inputs.get(key),candidate=True,extrapolate=True)
            if reason == 'applied':
                return delta,'nine_point_candidate:'+key+':applied'
            reasons.append(key+'='+reason)
        return np.zeros(2),'nine_point_candidate:base_fallback:'+','.join(reasons)

    def as_dict(self):
        result=super().as_dict()
        result.update(schema='nine-point-candidate-v1',
            policy='nine-point fit; bounded extrapolation for changed postures; invalid inputs fall back')
        return result
