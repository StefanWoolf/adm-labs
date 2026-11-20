"""Пути, константы и загрузка данных. Запуск `python -m src.data` скачивает датасет в data/raw."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "titanic.csv"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
TARGET = "survived"
NUM_COLS = ["age", "sibsp", "parch", "fare"]
CAT_COLS = ["pclass", "sex", "embarked"]


def download() -> None:
    import seaborn as sns

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    sns.load_dataset("titanic").to_csv(RAW_PATH, index=False)
    print(f"saved {RAW_PATH}")


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(RAW_PATH)
    return df[NUM_COLS + CAT_COLS], df[TARGET]


if __name__ == "__main__":
    download()
