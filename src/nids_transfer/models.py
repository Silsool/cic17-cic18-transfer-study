"""Baseline factory extracted from the original ml_models.py.

The two supported constructors preserve the original defaults. Experiment-specific
resource/solver overrides remain in BinaryStudy.fit_model. Neural models are not
part of this study's current baseline.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_cleanlab_model(name, num_classes, seed, verbose=False):
    if name == 'logreg':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == 'rf':
        return RandomForestClassifier(n_estimators=200, random_state=seed)
    raise ValueError(f'Unsupported baseline model: {name}')
