"""Кастомные шаги пайплайнов."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


class MissingRowFlag(BaseEstimator, TransformerMixin):
    """Лаба 1: добавляет бинарный признак has_missing (1, если в строке есть хотя бы один пропуск).

    Ставится ПЕРЕД импутацией, иначе пропусков уже не будет.
    Ничего не обучает, поэтому fit просто возвращает self.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["has_missing"] = X.isna().any(axis=1).astype(int)
        return X


class OutlierRemover(BaseEstimator):
    """Лаба 2: удаляет выбросы через IsolationForest только из обучающей выборки.

    Метода transform нет, есть fit_resample, как у сэмплеров imblearn.
    imblearn.Pipeline вызывает fit_resample только во время fit, а при predict
    этот шаг пропускает. Поэтому тестовые объекты никогда не удаляются.
    Обычный sklearn.Pipeline так не умеет: transform обязан вернуть столько же строк, сколько получил.
    """

    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state

    def fit_resample(self, X, y):
        self.detector_ = IsolationForest(contamination=self.contamination,
                                         random_state=self.random_state)
        inliers = self.detector_.fit_predict(X) == 1  # 1 = норма, -1 = выброс
        return X[inliers], np.asarray(y)[inliers]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Лаба 2: ручная генерация признаков (используется через FunctionTransformer)."""
    X = X.copy()
    X["family_size"] = X["sibsp"] + X["parch"] + 1
    X["fare_per_person"] = X["fare"] / X["family_size"]
    return X


GEN_COLS = ["family_size", "fare_per_person"]
