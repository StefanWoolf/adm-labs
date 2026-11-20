"""Повторяющиеся операции: построение пайплайна, кросс-валидация, оценка, подбор параметров, сохранение."""
import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (включает IterativeImputer)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

from src.data import CAT_COLS, MODELS_DIR, NUM_COLS, RANDOM_STATE
from src.transformers import GEN_COLS, add_features

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_pipeline(num_imputer, gen_features=False, outliers="passthrough", select="passthrough"):
    """Лаба 2: полный пайплайн. Ненужные шаги отключаются значением 'passthrough'.

    Числа: StandardScaler -> импутер (KNN/Iterative чувствительны к масштабу, поэтому
    масштабируем ДО импутации; StandardScaler игнорирует NaN при fit) -> [полиномы].
    Категории: мода -> OneHot.
    """
    num_steps = [("scale", StandardScaler()), ("impute", num_imputer)]
    if gen_features:
        num_steps += [("poly", PolynomialFeatures(degree=2, include_bias=False)),
                      ("scale_poly", StandardScaler())]
    num_cols = NUM_COLS + (GEN_COLS if gen_features else [])

    prep = ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         CAT_COLS),
    ])
    return ImbPipeline([
        ("gen", FunctionTransformer(add_features) if gen_features else "passthrough"),
        ("prep", prep),
        ("outliers", outliers),
        ("select", select),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def cv_auc(model, X, y):
    """Средний ROC-AUC на 5 стратифицированных фолдах."""
    scores = cross_val_score(model, X, y, cv=CV, scoring="roc_auc")
    return scores.mean(), scores.std()


def evaluate(model, X_test, y_test):
    """Метрики на отложенной выборке."""
    proba = model.predict_proba(X_test)[:, 1]
    return {"accuracy": accuracy_score(y_test, proba >= 0.5), "roc_auc": roc_auc_score(y_test, proba)}


def grid_search(model, param_grid, X, y):
    gs = GridSearchCV(model, param_grid, cv=CV, scoring="roc_auc", n_jobs=-1).fit(X, y)
    print("best params:", gs.best_params_)
    print(f"best CV ROC-AUC: {gs.best_score_:.4f}")
    return gs


def save_model(model, name):
    path = MODELS_DIR / name
    joblib.dump(model, path)
    print(f"saved {path}")
