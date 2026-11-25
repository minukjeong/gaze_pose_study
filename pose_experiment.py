"""Separate posture collection and explicitly labeled candidate-on evaluation."""
import numpy as np
from dynamic_gaze import DynamicGazeCalibration


class PoseExperimentCalibration(DynamicGazeCalibration):
    def __init__(self,base,screen):
        super().__init__(base,screen)
        self.experiment_enabled=False

    def enable_experiment(self):
        if not self.finished:
            raise ValueError('Complete separate posture adaptation and validation first')
        routes=[key for key,m in self.models.items() if m.fitted]
        if not routes:
            raise ValueError('No posture model could be fitted; repeat posture collection with visible movement')
        self.experiment_enabled=True
        return routes

    def predict(self,raw,inputs):
        if not self.experiment_enabled:
            return super().predict(raw,inputs)
        reasons=[]
        for key in ('head_shoulders','head'):
            model=self.models[key]
            delta,reason=model.correction(raw,inputs.get(key),candidate=True)
            if reason=='applied':return delta,'experiment:'+key+':applied'
            reasons.append(key+'='+reason)
        return np.zeros(2),'experiment:base_fallback:'+','.join(reasons)

    def as_dict(self):
        data=super().as_dict()
        data.update(schema='pose-candidate-evaluation-v1',experiment_enabled=self.experiment_enabled,
                    policy='candidate-on; retain original validation results and support checks',
                    fitted_routes=[key for key,m in self.models.items() if m.fitted])
        return data
