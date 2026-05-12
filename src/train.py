"""
Pipeline d'entraînement complète :
  1. Forecasting par paramètre (XGBoost Regressor, horizon j+7 et j+30)
  2. Classification de défaillance globale (Random Forest + Gradient Boosting)
  3. Validation croisée temporelle (TimeSeriesSplit)
  4. Sauvegarde des modèles et des métriques
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Paramètres à prévoir (forecasting individuel)
FORECAST_TARGETS = [
    "top_oil_temp",
    "hot_spot_temp",
    "H2",
    "CO",
    "C2H4",
    "CH4",
    "moisture_ppm",
]

# Horizons de prévision (en heures)
FORECAST_HORIZONS = {
    "7d":  7  * 24,   # 168 h
    "30d": 30 * 24,   # 720 h
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _save_model(obj: Any, name: str) -> Path:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Modèle sauvegardé → %s", path)
    return path


def _load_model(name: str) -> Any:
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Préparation des matrices X, y pour le forecasting
# ---------------------------------------------------------------------------

def _make_forecast_dataset(
    df: pd.DataFrame,
    target: str,
    horizon: int,
    feature_cols: list[str],
    n_lags: int = 24,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Crée (X, y) pour la prédiction de `target` à horizon heures.
    Features = lags du target + autres colonnes numériques actuelles.
    """
    df_work = df.copy()

    # Lags du target
    for lag in range(1, n_lags + 1):
        df_work[f"{target}_lag{lag}"] = df_work[target].shift(lag)

    # Cible décalée vers l'avenir
    df_work["_target"] = df_work[target].shift(-horizon)

    # Colonnes à utiliser
    lag_cols  = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
    feat_cols = [c for c in feature_cols if c in df_work.columns and c != target]
    all_cols  = feat_cols + lag_cols

    df_work = df_work.dropna(subset=all_cols + ["_target"])

    X = df_work[all_cols].values
    y = df_work["_target"].values
    idx = df_work.index

    return X, y, idx


# ---------------------------------------------------------------------------
# Entraînement Forecasting (XGBoost Regressor)
# ---------------------------------------------------------------------------

class ForecastTrainer:
    """
    Entraîne un modèle XGBoost Regressor par paramètre et par horizon.
    Effectue une validation croisée temporelle (5 folds).
    """

    def __init__(self, n_splits: int = 5, n_lags: int = 24):
        self.n_splits = n_splits
        self.n_lags   = n_lags
        self.models_  : dict[str, dict[str, Any]] = {}
        self.scalers_ : dict[str, StandardScaler]  = {}
        self.metrics_ : dict[str, dict[str, Any]]  = {}

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "ForecastTrainer":
        logger.info("=== Entraînement des modèles de forecasting ===")

        for target in FORECAST_TARGETS:
            if target not in df.columns:
                logger.warning("Colonne %s absente — sautée.", target)
                continue

            self.models_[target] = {}
            self.metrics_[target] = {}

            for horizon_label, horizon_h in FORECAST_HORIZONS.items():
                logger.info("  → %s | horizon %s", target, horizon_label)

                try:
                    X, y, _ = _make_forecast_dataset(
                        df, target, horizon_h, feature_cols, self.n_lags
                    )
                except Exception as exc:
                    logger.error("Erreur dataset %s/%s : %s", target, horizon_label, exc)
                    continue

                if len(X) < self.n_splits * 10:
                    logger.warning("Pas assez de données pour %s/%s.", target, horizon_label)
                    continue

                # Scaling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers_[f"{target}_{horizon_label}"] = scaler

                # Validation croisée temporelle
                tscv = TimeSeriesSplit(n_splits=self.n_splits)
                cv_rmse, cv_mae = [], []

                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_tr, y_val = y[train_idx],         y[val_idx]

                    model_cv = XGBRegressor(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0,
                    )
                    model_cv.fit(X_tr, y_tr)
                    preds = model_cv.predict(X_val)
                    cv_rmse.append(_safe_rmse(y_val, preds))
                    cv_mae.append(mean_absolute_error(y_val, preds))

                # Entraînement final sur tout le dataset
                model_final = XGBRegressor(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
                model_final.fit(X_scaled, y)

                self.models_[target][horizon_label] = model_final
                self.metrics_[target][horizon_label] = {
                    "cv_rmse_mean": float(np.mean(cv_rmse)),
                    "cv_rmse_std":  float(np.std(cv_rmse)),
                    "cv_mae_mean":  float(np.mean(cv_mae)),
                    "n_samples":    int(len(X)),
                }

                logger.info(
                    "    RMSE CV = %.3f ± %.3f | MAE CV = %.3f",
                    np.mean(cv_rmse), np.std(cv_rmse), np.mean(cv_mae),
                )

        return self

    def save(self) -> None:
        for target, horizons in self.models_.items():
            for hor, model in horizons.items():
                _save_model(model, f"forecast_{target}_{hor}")
        for key, scaler in self.scalers_.items():
            _save_model(scaler, f"scaler_forecast_{key}")

        metrics_path = MODELS_DIR / "forecast_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_, f, indent=2)
        logger.info("Métriques de forecasting → %s", metrics_path)


# ---------------------------------------------------------------------------
# Entraînement Modèle Global (Classification de défaillance)
# ---------------------------------------------------------------------------

class HealthClassifierTrainer:
    """
    Entraîne Random Forest + Gradient Boosting pour prédire la probabilité
    de défaillance dans les 30 prochains jours.
    Effectue une validation croisée temporelle.
    """

    LABEL_COL = "failure_label"

    def __init__(self, n_splits: int = 5):
        self.n_splits   = n_splits
        self.model_rf_  : RandomForestClassifier | None          = None
        self.model_gb_  : GradientBoostingClassifier | None      = None
        self.model_xgb_ : XGBClassifier | None                   = None
        self.scaler_    : StandardScaler | None                   = None
        self.feature_cols_ : list[str]                           = []
        self.metrics_   : dict[str, Any]                          = {}

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "HealthClassifierTrainer":
        logger.info("=== Entraînement du classifieur de défaillance ===")

        if self.LABEL_COL not in df.columns:
            raise ValueError(f"Colonne cible '{self.LABEL_COL}' absente du DataFrame.")

        feat_cols = [c for c in feature_cols if c in df.columns and c != self.LABEL_COL]
        self.feature_cols_ = feat_cols

        df_work = df[feat_cols + [self.LABEL_COL]].dropna()
        X = df_work[feat_cols].values
        y = df_work[self.LABEL_COL].values.astype(int)

        logger.info(
            "Dataset classification : %d samples | positifs=%.1f%%",
            len(X), y.mean() * 100,
        )

        # Scaling
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # ---- Random Forest ----
        rf_f1, rf_auc = [], []
        for train_idx, val_idx in tscv.split(X_scaled):
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_scaled[train_idx], y[train_idx])
            preds  = rf.predict(X_scaled[val_idx])
            probas = rf.predict_proba(X_scaled[val_idx])[:, 1]
            rf_f1.append(f1_score(y[val_idx], preds, zero_division=0))
            try:
                rf_auc.append(roc_auc_score(y[val_idx], probas))
            except Exception:
                pass

        # ---- Gradient Boosting ----
        gb_f1, gb_auc = [], []
        for train_idx, val_idx in tscv.split(X_scaled):
            gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
            gb.fit(X_scaled[train_idx], y[train_idx])
            preds  = gb.predict(X_scaled[val_idx])
            probas = gb.predict_proba(X_scaled[val_idx])[:, 1]
            gb_f1.append(f1_score(y[val_idx], preds, zero_division=0))
            try:
                gb_auc.append(roc_auc_score(y[val_idx], probas))
            except Exception:
                pass

        # ---- XGBoost Classifier ----
        xgb_f1, xgb_auc = [], []
        for train_idx, val_idx in tscv.split(X_scaled):
            xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, use_label_encoder=False,
                                 eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
            xgb.fit(X_scaled[train_idx], y[train_idx])
            preds  = xgb.predict(X_scaled[val_idx])
            probas = xgb.predict_proba(X_scaled[val_idx])[:, 1]
            xgb_f1.append(f1_score(y[val_idx], preds, zero_division=0))
            try:
                xgb_auc.append(roc_auc_score(y[val_idx], probas))
            except Exception:
                pass

        # Modèles finaux sur tout le dataset
        self.model_rf_ = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
        self.model_rf_.fit(X_scaled, y)

        self.model_gb_ = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
        self.model_gb_.fit(X_scaled, y)

        self.model_xgb_ = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, use_label_encoder=False,
                                          eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
        self.model_xgb_.fit(X_scaled, y)

        # Métriques finales sur le dernier fold (test set)
        final_split = list(tscv.split(X_scaled))[-1]
        _, test_idx = final_split
        X_test, y_test = X_scaled[test_idx], y[test_idx]

        best_model = self.model_xgb_  # XGBoost généralement meilleur
        y_pred  = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        self.metrics_ = {
            "rf_f1_cv":   float(np.mean(rf_f1)),
            "rf_auc_cv":  float(np.mean(rf_auc)) if rf_auc else None,
            "gb_f1_cv":   float(np.mean(gb_f1)),
            "gb_auc_cv":  float(np.mean(gb_auc)) if gb_auc else None,
            "xgb_f1_cv":  float(np.mean(xgb_f1)),
            "xgb_auc_cv": float(np.mean(xgb_auc)) if xgb_auc else None,
            "final_test_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix":  confusion_matrix(y_test, y_pred).tolist(),
        }

        logger.info("RF  → F1 CV = %.3f | AUC CV = %.3f", np.mean(rf_f1), np.mean(rf_auc) if rf_auc else 0)
        logger.info("GB  → F1 CV = %.3f | AUC CV = %.3f", np.mean(gb_f1), np.mean(gb_auc) if gb_auc else 0)
        logger.info("XGB → F1 CV = %.3f | AUC CV = %.3f", np.mean(xgb_f1), np.mean(xgb_auc) if xgb_auc else 0)
        logger.info("\n%s", classification_report(y_test, y_pred))

        return self

    def save(self) -> None:
        _save_model(self.model_rf_,  "classifier_rf")
        _save_model(self.model_gb_,  "classifier_gb")
        _save_model(self.model_xgb_, "classifier_xgb")
        _save_model(self.scaler_,    "scaler_classifier")
        _save_model(self.feature_cols_, "feature_cols")

        metrics_path = MODELS_DIR / "classifier_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_, f, indent=2)
        logger.info("Métriques du classifieur → %s", metrics_path)


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------

def run_training_pipeline(data_path: str = "data/raw/transformer_data.csv") -> None:
    """
    Exécute la pipeline complète :
    1. Chargement + nettoyage des données
    2. Feature engineering
    3. Entraînement forecasting
    4. Entraînement classifieur de défaillance
    5. Sauvegarde des modèles et métriques
    """
    from src.data_loader import load_and_prepare
    from src.features import build_features, get_feature_columns

    logger.info("Chargement des données depuis %s …", data_path)
    df = load_and_prepare(data_path)

    logger.info("Feature engineering …")
    df_feat = build_features(df)

    feature_cols = get_feature_columns(df_feat, exclude_targets=True)
    logger.info("%d features disponibles.", len(feature_cols))

    # ---- Forecasting ----
    forecast_trainer = ForecastTrainer(n_splits=5, n_lags=24)
    forecast_trainer.fit(df_feat, feature_cols)
    forecast_trainer.save()

    # ---- Classification ----
    clf_trainer = HealthClassifierTrainer(n_splits=5)
    clf_trainer.fit(df_feat, feature_cols + ["failure_label"])
    clf_trainer.save()

    logger.info("Pipeline d'entraînement terminée avec succès.")
    _print_summary()


def _print_summary() -> None:
    """Affiche un résumé des métriques sauvegardées."""
    for fname in ["forecast_metrics.json", "classifier_metrics.json"]:
        path = MODELS_DIR / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"\n=== {fname} ===")
            print(json.dumps(data, indent=2)[:2000])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_training_pipeline()
