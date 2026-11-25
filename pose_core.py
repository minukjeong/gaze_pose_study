"""Camera-independent study math. No fitted parameters or human results shipped."""
import json
from pathlib import Path
import numpy as np

SCHEMA = 1
JOINTS = (0, 11, 12, 13, 14, 15, 16, 23, 24)


def pose_features(world, image, rotation, center, threshold=0.35):
    """world: 33x3; image: 33x4 (normalized xy,z,visibility).

    Face rotation/position and body coordinates are separate feature groups;
    their coordinate systems are never treated as one metric skeleton.
    """
    world, image = np.asarray(world), np.asarray(image)
    ids = np.array(JOINTS)
    good = (image[ids, 3] >= threshold) & np.isfinite(world[ids]).all(axis=1)
    good &= np.isfinite(image[ids, :2]).all(axis=1)
    # MediaPipe can reliably place a shoulder just outside a tight webcam crop.
    # Permit a small margin while rejecting clearly extrapolated joints.
    good &= ((image[ids, :2] >= -0.15) & (image[ids, :2] <= 1.15)).all(axis=1)
    if not good[1:3].all():
        return None
    origin = (world[11] + world[12]) / 2
    scale = np.linalg.norm(world[11] - world[12])
    image_scale = np.linalg.norm(image[11, :2] - image[12, :2])
    if scale < 1e-4 or image_scale < 0.03:
        return None
    rel = (world[ids] - origin) / scale
    rel[~good] = 0
    visibility = np.where(good, image[ids, 3], 0)
    # Rotation matrix features avoid Euler-angle wraparound.
    feature = np.r_[np.asarray(rotation)[:, :2].ravel(), np.asarray(center).ravel(),
                    rel.ravel(), visibility, good.astype(float),
                    (image[11, :2] + image[12, :2]) / 2, image_scale]
    return feature if np.isfinite(feature).all() else None


def standard_fit(x):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1
    return mean, scale


def poly(x):
    x = np.atleast_2d(x)
    a, b = x[:, 0], x[:, 1]
    return np.c_[np.ones(len(x)), a, b, a*a, a*b, b*b]


def ridge(x, y, alpha):
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def fusion_design(g, z):
    return np.c_[np.ones(len(z)), z, z*g[:, :1], z*g[:, 1:2]]


class PoseGazeModel:
    def __init__(self, data):
        self.data = data

    @classmethod
    def fit(cls, trials, metadata, dimension=8, alpha=10.0):
        if any(t['phase'] != 'train' for t in trials):
            raise ValueError('Only training trials may fit any model component.')
        if len(trials) < 18 or len({tuple(t['target']) for t in trials}) < 6:
            raise ValueError('Need >=18 accepted trials and >=6 distinct targets.')
        raw = np.array([t['raw'] for t in trials])
        target = np.array([t['target'] for t in trials])
        gm, gs = standard_fit(raw)
        g = (raw-gm)/gs
        baseline = ridge(poly(g), target, 0.1)
        eligible = [i for i,t in enumerate(trials) if t.get('pose') is not None]
        if len(eligible) < 18:
            raise ValueError('Need >=18 accepted trials with valid shoulders/pose.')
        p = np.array([trials[i]['pose'] for i in eligible])
        pm, ps = standard_fit(p)
        standardized = (p-pm)/ps
        _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
        rank = int((singular > max(singular[0]*1e-8, 1e-8)).sum())
        if rank == 0:
            raise ValueError('No posture variation available for PCA.')
        k = min(dimension, rank, max(1, len(p)//6))
        components = vt[:k]
        zscale = np.maximum((standardized @ components.T).std(axis=0), 1e-6)
        z = standardized @ components.T / zscale
        residual = target[eligible] - poly(g[eligible]) @ baseline
        correction = ridge(fusion_design(g[eligible], z), residual, alpha)
        # Automatically discover personal posture profiles in embedding space.
        cluster_count = min(3, max(1, len(z)//12))
        centers_array = _kmeans(z, cluster_count)
        assignment = np.argmin(np.linalg.norm(z[:, None, :] - centers_array[None, :, :], axis=2), axis=1)
        centers = {f'cluster_{i+1}': center for i,center in enumerate(centers_array)}
        distances = [np.linalg.norm(row-centers_array[label]) for row,label in zip(z,assignment)]
        radius = max(1.0, float(np.quantile(distances, .95))*1.5)
        data = dict(schema=SCHEMA, metadata=metadata, gaze_mean=gm, gaze_scale=gs,
                    baseline=baseline, pose_mean=pm, pose_scale=ps,
                    components=components, embedding_scale=zscale, correction=correction,
                    centers=centers, profile_radius=radius, dimension=k, alpha=alpha,
                    train_trials=len(trials), pose_trials=len(p))
        return cls(data)

    def predict(self, raw, pose=None):
        d = self.data
        g = (np.atleast_2d(raw)-d['gaze_mean'])/d['gaze_scale']
        base = (poly(g) @ d['baseline'])[0]
        out = dict(baseline=base, fused=base.copy(), embedding=None,
                   profile='unknown', fusion_used=False, reason='pose_missing')
        if pose is None:
            return out
        p = np.asarray(pose)
        if p.shape != np.asarray(d['pose_mean']).shape or not np.isfinite(p).all():
            return out
        z = ((p-d['pose_mean'])/d['pose_scale']) @ np.asarray(d['components']).T / d['embedding_scale']
        distances = {name: float(np.linalg.norm(z-c)) for name,c in d['centers'].items()}
        nearest = min(distances, key=distances.get)
        out['embedding'] = z
        # Training-only heuristic, not a calibrated probability of correctness.
        if distances[nearest] > d['profile_radius']:
            out['reason'] = 'outside_training_postures'
            return out
        out['profile'] = nearest
        delta = (fusion_design(g, z[None]) @ d['correction'])[0]
        if not np.isfinite(delta).all():
            out['reason'] = 'nonfinite_correction'
            return out
        out.update(fused=base+delta, fusion_used=True, reason='applied')
        return out

    def save(self, path):
        Path(path).write_text(json.dumps(self.data, default=json_default, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        if d.get('schema') != SCHEMA:
            raise ValueError('Unsupported model schema')
        for key in ('gaze_mean','gaze_scale','baseline','pose_mean','pose_scale','components',
                    'embedding_scale','correction'):
            d[key] = np.array(d[key], dtype=float)
        d['centers'] = {k: np.array(v) for k,v in d['centers'].items()}
        return cls(d)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _kmeans(x, k, iterations=50):
    """Deterministic small k-means for personal posture profiles."""
    x = np.asarray(x, dtype=float)
    centers = [x[np.argmax(np.linalg.norm(x-x.mean(axis=0), axis=1))]]
    while len(centers) < k:
        d = np.min(np.linalg.norm(x[:, None, :]-np.array(centers)[None, :, :], axis=2), axis=1)
        centers.append(x[np.argmax(d)])
    centers = np.array(centers)
    for _ in range(iterations):
        labels = np.argmin(np.linalg.norm(x[:, None, :]-centers[None, :, :], axis=2), axis=1)
        updated = np.array([x[labels==i].mean(axis=0) if np.any(labels==i) else centers[i] for i in range(k)])
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


def aggregate_trial(rows, phase, trial_id, posture, comfort, target, min_samples=8):
    valid = [r for r in rows if r.get('raw') is not None]
    if len(valid) < min_samples or len(valid)/max(1,len(rows)) < .6:
        return None
    posed = [r['pose'] for r in valid if r.get('pose') is not None]
    return dict(phase=phase, trial_id=trial_id, posture=posture, comfort_self_report=comfort,
                target=target, raw=np.mean([r['raw'] for r in valid],axis=0),
                pose=np.mean(posed,axis=0) if len(posed)>=min_samples and len(posed)/len(valid)>=.6 else None,
                samples=len(valid), total_samples=len(rows))


def evaluate_rows(rows, width, height):
    """Trial-balanced frame errors; unbounded predictions, no clipping/smoothing."""
    trials = {}
    for r in rows:
        if r.get('phase') != 'evaluate' or r.get('target') is None:
            continue
        key = (r['posture'],r['trial_id'])
        trials.setdefault(key, []).append(r)
    records = []
    scale = np.array([width,height])
    for (posture, trial), group in trials.items():
        valid = [r for r in group if r.get('baseline') is not None and r.get('fused') is not None]
        rec = dict(posture=posture, trial_id=trial, frames=len(group), valid_frames=len(valid))
        if valid:
            for method in ('baseline','fused'):
                errors = [float(np.linalg.norm((np.array(r[method])-r['target'])*scale)) for r in valid]
                rec[method+'_mean_px'] = float(np.mean(errors))
            rec['fusion_fraction'] = sum(r['fusion_used'] for r in valid)/len(valid)
        records.append(rec)
    accepted = [r for r in records if r['valid_frames']>=8 and r['valid_frames']/r['frames']>=.6]
    def summarize(items):
        good = [r for r in items if r in accepted]
        result = dict(total_trials=len(items), accepted_trials=len(good),
                      recorded_frames=sum(r['frames'] for r in items),
                      valid_frames=sum(r['valid_frames'] for r in items))
        for method in ('baseline','fused'):
            vals = [r[method+'_mean_px'] for r in good]
            result[method+'_mean_px'] = float(np.mean(vals)) if vals else None
        if good:
            result['improvement_px'] = result['baseline_mean_px']-result['fused_mean_px']
            result['improvement_percent'] = (100*result['improvement_px']/result['baseline_mean_px']
                                             if result['baseline_mean_px']>0 else None)
        return result
    return dict(metric='Mean frame Euclidean pixel error, equally weighted accepted trials',
                target_type='instructed fixation target; not measured eye-tracker ground truth',
                overall=summarize(records),
                by_posture={p:summarize([r for r in records if r['posture']==p]) for p in sorted({r['posture'] for r in records})},
                trials=records)
