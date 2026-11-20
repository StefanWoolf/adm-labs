"""Лаба 5 (вариант 9): синтетика, Spark, AutoGluon, MLflow.

    python -m src.lab5 check              проверить Java, PySpark, AutoGluon и MLflow (~1 мин)
    python -m src.lab5 generate [N]       сгенерировать data/raw/lab5_synth.parquet (N строк, по умолчанию 5 млн)
"""
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from src.data import RANDOM_STATE, ROOT

LAB5_RAW = ROOT / "data" / "raw" / "lab5_synth.parquet"
LABEL = "label"
FEATS = [f"feat_{i:02d}" for i in range(1, 30)]  # feat_00 превращён в amount
EXPERIMENT = "lab5_variant9_synthetic_financial"


def generate(n_rows: int = 5_000_000) -> None:
    """make_classification с 30 признаками + «финансовая» обвязка и намеренные дефекты для очистки в Spark.

    - feat_00 становится суммой операции amount = exp(feat_00/2 + 7) (логнормальная, как реальные суммы);
    - client_id (100 тыс. клиентов) и region (по клиенту) нужны для групповых признаков;
    - ts записан СТРОКОЙ, чтобы в Spark было что приводить к типу timestamp;
    - дефекты: 2% пропусков в amount, feat_01, feat_02 и 0.5% полных дублей строк.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sklearn.datasets import make_classification

    rng = np.random.default_rng(RANDOM_STATE)
    X, y = make_classification(n_samples=n_rows, n_features=30, n_informative=12, n_redundant=6,
                               weights=[0.8, 0.2], flip_y=0.02, class_sep=0.8, random_state=RANDOM_STATE)
    df = pd.DataFrame(X[:, 1:].astype("float32"), columns=FEATS)
    df.insert(0, "amount", np.round(np.exp(X[:, 0] / 2 + 7), 2).astype("float32"))
    del X
    client = rng.integers(0, 100_000, n_rows, dtype="int32")
    df["client_id"] = client
    df["region"] = pd.Categorical.from_codes(client % 10, [f"R{i}" for i in range(10)]).astype(str)
    seconds = rng.integers(0, 365 * 24 * 3600, n_rows)
    df["ts"] = (np.datetime64("2025-01-01T00:00:00") + seconds.astype("timedelta64[s]")).astype(str)
    df["ts"] = df["ts"].str.replace("T", " ")
    df[LABEL] = y.astype("int8")

    for col in ["amount", "feat_01", "feat_02"]:
        df.loc[rng.random(n_rows) < 0.02, col] = np.nan
    dups = df.sample(frac=0.005, random_state=RANDOM_STATE)
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)

    LAB5_RAW.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), LAB5_RAW, row_group_size=500_000)
    print(f"saved {LAB5_RAW}: {len(df):,} rows, {LAB5_RAW.stat().st_size / 2**20:.0f} MB")


def _java_major(java_home: str | None) -> int:
    java = os.path.join(java_home, "bin", "java") if java_home else "java"
    out = subprocess.run([java, "-version"], capture_output=True, text=True).stderr
    return int(re.search(r'version "(\d+)', out).group(1))


def check_java() -> str:
    """Spark берёт java из JAVA_HOME (если задан), иначе из PATH. Нужна 17 или 21:
    на Java 23+ Hadoop внутри Spark падает на Subject.getSubject (UnsupportedOperationException).

    Если подходящей нет (например, в PATH Java 26, а IDE не передала JAVA_HOME), ищем установленную
    JDK 21/17 в стандартных папках и выставляем JAVA_HOME для этого процесса — до старта JVM.
    """
    from glob import glob

    major = _java_major(os.environ.get("JAVA_HOME"))
    if major not in (17, 21):
        roots = [r"C:\Program Files\Eclipse Adoptium", r"C:\Program Files\Java", r"C:\Program Files\Microsoft"]
        found = [h for pat in ("jdk-21*", "jdk-17*") for r in roots for h in sorted(glob(os.path.join(r, pat)))]
        if not found:
            raise RuntimeError(f"нужна Java 17 или 21, найдена {major}. Установите Temurin 21 и задайте JAVA_HOME")
        os.environ["JAVA_HOME"] = found[0]  # 21 приоритетнее 17
        major = _java_major(found[0])
    return f"{major} ({os.environ.get('JAVA_HOME', 'PATH')})"


def get_spark(driver_memory: str = "4g"):
    """Локальный Spark на всех ядрах. На Windows воркерам явно указываем текущий python (иначе ищут `python3`)."""
    from pyspark.sql import SparkSession

    check_java()

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (SparkSession.builder.master("local[*]").appName("lab5")
            .config("spark.driver.memory", driver_memory)
            .config("spark.driver.host", "127.0.0.1")  # не зависеть от имени хоста/VPN-адаптеров
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "16")  # дефолт 200 — слишком много для одной машины
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")  # быстрый toPandas через Arrow
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def setup_mlflow(experiment: str = EXPERIMENT):
    """Локальный MLflow: метаданные в mlflow.db (SQLite), артефакты в mlruns/ в корне проекта."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


def metrics(y_true, proba, threshold: float = 0.5) -> dict:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

    pred = proba >= threshold
    return {"roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
            "accuracy": accuracy_score(y_true, pred), "f1": f1_score(y_true, pred)}


def check() -> None:
    """Быстрая проверка окружения: каждая строка OK или понятная ошибка."""
    import tempfile
    from pathlib import Path

    print(f"OK java {check_java()}")

    import autogluon.tabular
    import mlflow
    import pyspark
    from autogluon.tabular import TabularPredictor
    from sklearn.datasets import make_classification

    X, y = make_classification(2000, 10, random_state=RANDOM_STATE)
    pdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]).assign(label=y)
    # ignore_cleanup_errors: на Windows MLflow (SQLAlchemy) держит m.db открытым, удалить его сразу нельзя
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        pdf.to_parquet(Path(tmp) / "x.parquet")
        spark = get_spark("1g")
        n = spark.read.parquet((Path(tmp) / "x.parquet").as_posix()).groupBy("label").count().toPandas()["count"].sum()
        spark.stop()
        print(f"OK pyspark {pyspark.__version__} (read parquet + groupBy + toPandas, {n} rows)")

        pred = TabularPredictor("label", eval_metric="roc_auc", path=str(Path(tmp) / "ag"), verbosity=0)
        pred.fit(pdf, time_limit=30, hyperparameters={"GBM": {}})
        print(f"OK autogluon {autogluon.tabular.__version__} (fit LightGBM, val roc_auc {pred.leaderboard()['score_val'][0]:.3f})")

        mlflow.set_tracking_uri(f"sqlite:///{Path(tmp, 'm.db').as_posix()}")
        with mlflow.start_run():
            mlflow.log_metric("x", 1.0)
        mlflow.set_tracking_uri("")
        print(f"OK mlflow {mlflow.__version__}")
    print(f"python {sys.version.split()[0]} — окружение готово к лабе 5")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "generate":
        generate(int(sys.argv[2]) if len(sys.argv) > 2 else 5_000_000)
    else:
        check()
