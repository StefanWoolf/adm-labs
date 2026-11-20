"""Лаба 4: MLP на PyTorch для табличных данных. Категории подаются через nn.Embedding."""
import copy

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import CAT_COLS, NUM_COLS, RANDOM_STATE
from src.transformers import GEN_COLS, add_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM = NUM_COLS + GEN_COLS


def make_prep():
    """Числа: медиана -> StandardScaler. Категории: мода -> OrdinalEncoder (Label Encoding).

    На выходе массив: сначала len(NUM) числовых колонок, затем коды категорий.
    Неизвестная на test категория получает код -1; в to_tensors все коды сдвигаются на +1,
    так что индекс 0 в эмбеддинге зарезервирован под «неизвестно».
    """
    return make_pipeline(FunctionTransformer(add_features), ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), CAT_COLS),
    ]))


def cardinalities(prep) -> list[int]:
    """Число индексов для каждого эмбеддинга: категории из train + 1 под «неизвестно»."""
    enc = prep[-1].named_transformers_["cat"][-1]
    return [len(c) + 1 for c in enc.categories_]


def to_tensors(prep, X, y=None):
    """DataFrame -> (x_num float32, x_cat int64[, y float32])."""
    Z = prep.transform(X)
    k = len(NUM)
    out = [torch.tensor(Z[:, :k], dtype=torch.float32), torch.tensor(Z[:, k:] + 1, dtype=torch.long)]
    if y is not None:
        out.append(torch.tensor(np.asarray(y), dtype=torch.float32))
    return out


class TabularMLP(nn.Module):
    """[числа | эмбеддинги категорий] -> (Linear -> BatchNorm -> ReLU -> Dropout) x N -> Linear(1)."""

    def __init__(self, n_num, cards, hidden=(128, 64), dropout=0.3):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c, min(50, (c + 1) // 2)) for c in cards)
        d = n_num + sum(e.embedding_dim for e in self.embs)
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)  # логит; сигмоида внутри BCEWithLogitsLoss

    def hidden(self, x_num, x_cat):
        """Выход последнего скрытого слоя: скрытое представление объекта."""
        z = torch.cat([x_num, *(e(x_cat[:, i]) for i, e in enumerate(self.embs))], dim=1)
        return self.body(z)

    def forward(self, x_num, x_cat):
        return self.head(self.hidden(x_num, x_cat)).squeeze(1)


@torch.no_grad()
def predict_proba(model, x_num, x_cat) -> np.ndarray:
    model.eval()  # BatchNorm берёт накопленные статистики, Dropout выключен
    return torch.sigmoid(model(x_num.to(DEVICE), x_cat.to(DEVICE))).cpu().numpy()


@torch.no_grad()
def hidden_repr(model, x_num, x_cat) -> np.ndarray:
    model.eval()
    return model.hidden(x_num.to(DEVICE), x_cat.to(DEVICE)).cpu().numpy()


def fit_mlp(train, cards, val=None, hidden=(128, 64), dropout=0.3, lr=1e-3,
            batch_size=64, epochs=300, patience=20, seed=RANDOM_STATE):
    """Обучение Adam + BCEWithLogitsLoss.

    С val: ранняя остановка по val ROC-AUC (patience эпох без улучшения), возвращаются веса лучшей эпохи.
    Без val: ровно `epochs` эпох (финальное дообучение на всём train с числом эпох, найденным на валидации).
    Возвращает (model, history, best_epoch).
    """
    torch.manual_seed(seed)
    x_num, x_cat, y = train
    model = TabularMLP(x_num.shape[1], cards, hidden, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    # drop_last: батч из одного объекта ломает BatchNorm (дисперсия по одному значению не определена)
    loader = DataLoader(TensorDataset(x_num, x_cat, y), batch_size=batch_size, shuffle=True,
                        drop_last=True, generator=torch.Generator().manual_seed(seed))

    history, best_auc, best_epoch, best_state = [], -np.inf, epochs, None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xn, xc, yb in loader:
            xn, xc, yb = xn.to(DEVICE), xc.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xn, xc), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        row = {"epoch": epoch, "train_loss": total / (len(loader) * batch_size)}
        if val is not None:
            row["val_auc"] = roc_auc_score(val[2].numpy(), predict_proba(model, val[0], val[1]))
            if row["val_auc"] > best_auc:
                best_auc, best_epoch, best_state = row["val_auc"], epoch, copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= patience:
                history.append(row)
                break
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch
