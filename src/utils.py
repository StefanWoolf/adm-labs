"""Повторяющиеся операции: построение пайплайна, кросс-валидация, оценка, подбор параметров, сохранение."""
import joblib
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (включает IterativeImputer)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler

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


def tree_pipeline(model, impute=False):
    """Лаба 3: пайплайн для деревьев. Масштабирование и one-hot деревьям не нужны.

    family_size/fare_per_person -> категории в порядковые коды (OrdinalEncoder) -> модель.
    Бустинги сами обрабатывают NaN (на каждом сплите выбирают, куда отправлять пропуски),
    поэтому по умолчанию пропуски не заполняем. Для RandomForest impute=True (медиана/мода).
    Выход ColumnTransformer — DataFrame с именами колонок, чтобы SHAP подписывал признаки.
    """
    num = SimpleImputer(strategy="median") if impute else "passthrough"
    cat = [("impute", SimpleImputer(strategy="most_frequent"))] if impute else []
    cat.append(("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    prep = ColumnTransformer([
        ("num", num, NUM_COLS + GEN_COLS),
        ("cat", Pipeline(cat), CAT_COLS),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    return ImbPipeline([("gen", FunctionTransformer(add_features)), ("prep", prep), ("model", model)])


SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"}


def cv_report(models: dict, X, y) -> pd.DataFrame:
    """Лаба 3: CV (5 фолдов) для словаря {имя: модель}. ROC-AUC, PR-AUC, F1 и время обучения одного фолда."""
    rows = {}
    for name, model in models.items():
        r = cross_validate(model, X, y, cv=CV, scoring=SCORING)
        rows[name] = {m: r[f"test_{m}"].mean() for m in SCORING} | {
            "roc_auc_std": r["test_roc_auc"].std(), "fit_time_s": r["fit_time"].mean()}
    return pd.DataFrame(rows).T


def cv_auc(model, X, y):
    """Средний ROC-AUC на 5 стратифицированных фолдах."""
    scores = cross_val_score(model, X, y, cv=CV, scoring="roc_auc")
    return scores.mean(), scores.std()


def evaluate(model, X_test, y_test):
    """Метрики на отложенной выборке."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = proba >= 0.5
    return {"accuracy": accuracy_score(y_test, pred), "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba), "f1": f1_score(y_test, pred)}


def grid_search(model, param_grid, X, y):
    gs = GridSearchCV(model, param_grid, cv=CV, scoring="roc_auc", n_jobs=-1).fit(X, y)
    print("best params:", gs.best_params_)
    print(f"best CV ROC-AUC: {gs.best_score_:.4f}")
    return gs


def save_model(model, name):
    path = MODELS_DIR / name
    joblib.dump(model, path)
    print(f"saved {path}")
