"""
Module d'inférence : chargement des modèles entraînés et prédictions en temps réel.

Expose :
- predict_health(df)         → probabilité de défaillance + indice de santé
- predict_forecast(df, target, horizon) → valeurs futures d'un paramètre
- generate_alerts(result)    → liste d'alertes de maintenance
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")

# Seuils d'alerte (modifiables selon politique de maintenance)
ALERT_THRESHOLDS = {
    "health_index_critical": 40.0,   # % — défaillance imminente
    "health_index_warning":  65.0,   # % — surveiller de près
    "top_oil_temp_max":     100.0,   # °C
    "hot_spot_temp_max":    130.0,   # °C
    "H2_max":               150.0,   # ppm
    "C2H4_max":              50.0,   # ppm
    "CH4_max":              120.0,   # ppm
    "CO_max":               500.0,   # ppm
    "moisture_max":          30.0,   # ppm
    "vibration_max":          5.0,   # mm/s
}

# Horizons disponibles
HORIZON_MAP = {"7d": 7 * 24, "30d": 30 * 24}


# ---------------------------------------------------------------------------
# Helpers de chargement
# ---------------------------------------------------------------------------

def _load(name: str) -> Any:
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle '{name}' introuvable dans {MODELS_DIR}. "
            "Lancez d'abord src/train.py ou python -m src.train."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Classe principale d'inférence
# ---------------------------------------------------------------------------

class TransformerPredictor:
    """
    Charge les modèles une seule fois et expose des méthodes de prédiction.
    """

    def __init__(self):
        self._clf_model    = None
        self._clf_scaler   = None
        self._feature_cols = None
        self._forecast_models : dict[str, dict[str, Any]] = {}
        self._forecast_scalers: dict[str, Any]            = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Chargement lazy des modèles
    # ------------------------------------------------------------------
    def load_models(self) -> "TransformerPredictor":
        """Charge tous les artefacts sauvegardés dans models/."""
        logger.info("Chargement des modèles depuis %s …", MODELS_DIR)

        try:
            self._clf_model    = _load("classifier_xgb")
            self._clf_scaler   = _load("scaler_classifier")
            self._feature_cols = _load("feature_cols")
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Modèles de classification introuvables. "
                "Exécutez d'abord python -m src.train"
            ) from exc

        # Chargement des modèles de forecasting (optionnel — pas d'erreur si absent)
        from src.train import FORECAST_TARGETS, FORECAST_HORIZONS
        for target in FORECAST_TARGETS:
            self._forecast_models[target] = {}
            for hor in FORECAST_HORIZONS:
                try:
                    self._forecast_models[target][hor]   = _load(f"forecast_{target}_{hor}")
                    self._forecast_scalers[f"{target}_{hor}"] = _load(f"scaler_forecast_{target}_{hor}")
                except FileNotFoundError:
                    logger.debug("Modèle de forecast %s/%s absent.", target, hor)

        self._loaded = True
        logger.info("Modèles chargés avec succès.")
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_models()

    # ------------------------------------------------------------------
    # Prédiction santé globale
    # ------------------------------------------------------------------
    def predict_health(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit la probabilité de défaillance et l'indice de santé résiduel.

        Parameters
        ----------
        df : DataFrame avec features (sorti de build_features())

        Returns
        -------
        DataFrame avec colonnes ajoutées :
          - failure_proba   : probabilité de défaillance dans les 30 jours (0–1)
          - health_score    : indice de santé (0–100 %)
          - risk_level      : "CRITIQUE" / "ALERTE" / "NORMAL"
        """
        self._ensure_loaded()

        feat_cols = [c for c in self._feature_cols if c in df.columns]
        missing   = [c for c in self._feature_cols if c not in df.columns]
        if missing:
            logger.warning("%d features manquantes (seront mises à 0) : %s", len(missing), missing[:10])
            for c in missing:
                df[c] = 0.0

        X = df[feat_cols].fillna(0).values
        X_scaled = self._clf_scaler.transform(X)

        proba = self._clf_model.predict_proba(X_scaled)[:, 1]

        result = df.copy()
        result["failure_proba"] = proba
        result["health_score"]  = np.clip((1 - proba) * 100, 0, 100)
        result["risk_level"]    = pd.cut(
            result["health_score"],
            bins=[0, 40, 65, 100],
            labels=["CRITIQUE", "ALERTE", "NORMAL"],
            right=True,
        ).astype(str)

        logger.info(
            "Prédiction santé : %.1f%% CRITIQUE | %.1f%% ALERTE | %.1f%% NORMAL",
            (result["risk_level"] == "CRITIQUE").mean() * 100,
            (result["risk_level"] == "ALERTE").mean()   * 100,
            (result["risk_level"] == "NORMAL").mean()   * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Prédiction forecasting
    # ------------------------------------------------------------------
    def predict_forecast(
        self,
        df: pd.DataFrame,
        target: str,
        horizon: str = "7d",
        feature_cols: list[str] | None = None,
        n_lags: int = 24,
    ) -> np.ndarray:
        """
        Prédit les valeurs futures de `target` à l'horizon spécifié.

        Parameters
        ----------
        df          : DataFrame historique avec features
        target      : colonne à prédire (ex: 'top_oil_temp')
        horizon     : '7d' ou '30d'
        feature_cols: liste de features (si None, utilise _feature_cols)
        n_lags      : nombre de lags du target à inclure

        Returns
        -------
        np.ndarray de prédictions (une par ligne de df)
        """
        self._ensure_loaded()

        if target not in self._forecast_models or horizon not in self._forecast_models[target]:
            raise ValueError(
                f"Modèle de forecast pour '{target}/{horizon}' indisponible. "
                "Vérifiez que l'entraînement a inclus cette configuration."
            )

        model  = self._forecast_models[target][horizon]
        scaler = self._forecast_scalers[f"{target}_{horizon}"]

        if feature_cols is None:
            feature_cols = self._feature_cols or []

        # Reconstruction du dataset avec lags
        from src.train import _make_forecast_dataset
        try:
            X, y_true, idx = _make_forecast_dataset(
                df, target, HORIZON_MAP[horizon], feature_cols, n_lags
            )
        except Exception as exc:
            raise RuntimeError(f"Erreur lors de la création du dataset : {exc}") from exc

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        # Reconstruit un tableau aligné sur df.index (NaN pour les premières lignes)
        result = np.full(len(df), np.nan)
        # idx contient les indices du df_work après dropna
        pos = [df.index.get_loc(i) for i in idx if i in df.index]
        for i, p in enumerate(pos):
            if i < len(predictions):
                result[p] = predictions[i]

        return result

    # ------------------------------------------------------------------
    # Prédiction sur un seul instant (ligne unique)
    # ------------------------------------------------------------------
    def predict_single(self, row: dict) -> dict:
        """
        Prédit l'état de santé pour un relevé unique (dictionnaire de valeurs).

        Parameters
        ----------
        row : dict avec les valeurs brutes du transformateur

        Returns
        -------
        dict avec failure_proba, health_score, risk_level, alerts
        """
        self._ensure_loaded()

        from src.features import build_features

        df_single = pd.DataFrame([row])
        if "timestamp" in df_single.columns:
            df_single = df_single.set_index("timestamp")
        elif not isinstance(df_single.index, pd.DatetimeIndex):
            df_single.index = pd.date_range("2024-01-01", periods=1, freq="1h")

        try:
            df_feat = build_features(df_single)
        except Exception as exc:
            logger.warning("Feature engineering partiel : %s", exc)
            df_feat = df_single.copy()

        result_df = self.predict_health(df_feat)
        row_out = result_df.iloc[0]

        return {
            "failure_proba": float(row_out.get("failure_proba", 0)),
            "health_score":  float(row_out.get("health_score", 100)),
            "risk_level":    str(row_out.get("risk_level", "NORMAL")),
            "alerts":        generate_alerts(row),
        }

    # ------------------------------------------------------------------
    # Rapport complet
    # ------------------------------------------------------------------
    def full_report(self, df: pd.DataFrame) -> dict:
        """
        Génère un rapport synthétique : santé courante, tendances, alertes actives.
        """
        result_df = self.predict_health(df)
        latest    = result_df.iloc[-1]

        alerts = []
        for col, thresh in ALERT_THRESHOLDS.items():
            if "_max" in col:
                raw_col = col.replace("_max", "")
                if raw_col in df.columns and df[raw_col].iloc[-1] > thresh:
                    alerts.append({
                        "parameter": raw_col,
                        "value":     float(df[raw_col].iloc[-1]),
                        "threshold": thresh,
                        "severity":  "CRITIQUE" if df[raw_col].iloc[-1] > thresh * 1.3 else "ALERTE",
                    })

        return {
            "timestamp":     str(result_df.index[-1]),
            "health_score":  float(latest["health_score"]),
            "failure_proba": float(latest["failure_proba"]),
            "risk_level":    str(latest["risk_level"]),
            "active_alerts": alerts,
            "n_samples":     len(df),
        }


# ---------------------------------------------------------------------------
# Génération d'alertes basée sur les seuils
# ---------------------------------------------------------------------------

def generate_alerts(measurements: dict | pd.Series) -> list[dict]:
    """
    Compare les mesures aux seuils et retourne une liste d'alertes.
    """
    if isinstance(measurements, pd.Series):
        measurements = measurements.to_dict()

    alerts = []
    for col, thresh in ALERT_THRESHOLDS.items():
        if "_max" not in col:
            continue
        raw_col = col.replace("_max", "")
        val = measurements.get(raw_col)
        if val is None:
            continue
        if val > thresh:
            severity = "CRITIQUE" if val > thresh * 1.3 else "ALERTE"
            alerts.append({
                "parameter": raw_col,
                "value":     round(float(val), 2),
                "threshold": thresh,
                "severity":  severity,
                "message":   f"{raw_col} = {val:.1f} > seuil {thresh} ({severity})",
            })

    return alerts


# ---------------------------------------------------------------------------
# Chargement des métriques pour affichage dans le dashboard
# ---------------------------------------------------------------------------

def load_metrics() -> dict:
    """Charge les métriques de performance depuis les fichiers JSON."""
    metrics = {}
    for fname in ["forecast_metrics.json", "classifier_metrics.json"]:
        path = MODELS_DIR / fname
        if path.exists():
            with open(path) as f:
                metrics[fname.replace(".json", "")] = json.load(f)
    return metrics


# ---------------------------------------------------------------------------
# Singleton global pour l'application Streamlit
# ---------------------------------------------------------------------------

_predictor_instance: TransformerPredictor | None = None


def get_predictor() -> TransformerPredictor:
    """Retourne l'instance singleton du prédicteur (chargement une seule fois)."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = TransformerPredictor()
        _predictor_instance.load_models()
    return _predictor_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_and_prepare
    from src.features import build_features

    df = load_and_prepare("data/raw/transformer_data.csv")
    df_feat = build_features(df)

    predictor = TransformerPredictor().load_models()
    result = predictor.predict_health(df_feat)

    print("\n=== Dernières prédictions ===")
    print(result[["health_score", "failure_proba", "risk_level"]].tail(10))

    report = predictor.full_report(df_feat)
    print("\n=== Rapport complet ===")
    print(json.dumps(report, indent=2))
