"""
Feature Engineering pour la maintenance prédictive de transformateur.

Calcule :
- Moyennes mobiles (7 j, 30 j)
- Taux de variation (dérivées des gaz dissous)
- Ratios de Duval (Triangle de Duval)
- Ratios de Rogers (diagnostic de défaut)
- Features de charge et thermiques
- Encodage temporel
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Colonnes brutes des gaz dissous
GAS_COLS = ["H2", "CO", "C2H4", "CH4"]
# Colonnes cibles pour les moyennes mobiles
ROLLING_COLS = ["top_oil_temp", "hot_spot_temp", "load_kva", "H2", "CO", "C2H4", "CH4", "moisture_ppm", "vibration_mms"]


# ---------------------------------------------------------------------------
# Fonctions de feature engineering
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame, windows_hours: list[int] | None = None) -> pd.DataFrame:
    """
    Ajoute les moyennes mobiles et écarts-types roulants pour les paramètres clés.
    Par défaut : fenêtres de 7 jours (168 h) et 30 jours (720 h).
    """
    if windows_hours is None:
        windows_hours = [168, 720]  # 7 jours, 30 jours

    df = df.copy()
    for w in windows_hours:
        label = f"{w}h"
        for col in ROLLING_COLS:
            if col not in df.columns:
                continue
            df[f"{col}_mean_{label}"]  = df[col].rolling(window=w, min_periods=max(1, w // 4)).mean()
            df[f"{col}_std_{label}"]   = df[col].rolling(window=w, min_periods=max(1, w // 4)).std()
            df[f"{col}_max_{label}"]   = df[col].rolling(window=w, min_periods=max(1, w // 4)).max()

    logger.debug("Ajout des features roulantes : %d nouvelles colonnes.", len(windows_hours) * len(ROLLING_COLS) * 3)
    return df


def add_rate_of_change(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """
    Taux de variation (diff / dt) pour les gaz dissous et températures.
    Détecte une accélération de la dégradation.
    """
    if periods is None:
        periods = [1, 24, 168]  # 1 h, 24 h, 7 jours

    df = df.copy()
    roc_cols = GAS_COLS + ["top_oil_temp", "hot_spot_temp", "moisture_ppm"]
    for col in roc_cols:
        if col not in df.columns:
            continue
        for p in periods:
            df[f"{col}_roc_{p}h"] = df[col].diff(periods=p)

    logger.debug("Taux de variation calculés.")
    return df


def add_duval_triangle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implémente le Triangle de Duval (IEC 60599) pour classer les défauts.
    Utilise les proportions relatives CH4, C2H4, C2H2 (ici approximé via CH4 + C2H4).

    Zones Duval :
    PD  = décharges partielles
    T1  = défaut thermique < 300 °C
    T2  = défaut thermique 300–700 °C
    T3  = défaut thermique > 700 °C
    D1  = décharge de faible énergie
    D2  = décharge de haute énergie
    DT  = défaut thermique + électrique mixte

    Note : Sans C2H2, nous utilisons une approximation basée sur CH4 et C2H4.
    """
    df = df.copy()

    # Somme des gaz du triangle (approx sans C2H2)
    total = df["CH4"] + df["C2H4"] + df["H2"].clip(lower=0)
    eps = 1e-9

    pct_ch4  = df["CH4"]  / (total + eps) * 100
    pct_c2h4 = df["C2H4"] / (total + eps) * 100
    pct_h2   = df["H2"]   / (total + eps) * 100

    df["duval_pct_CH4"]  = pct_ch4
    df["duval_pct_C2H4"] = pct_c2h4
    df["duval_pct_H2"]   = pct_h2

    # Classification simplifiée par zones
    conditions = [
        (pct_h2 > 98),                                        # PD
        (pct_ch4 > 80),                                       # T1
        (pct_c2h4 > 20) & (pct_c2h4 <= 50) & (pct_ch4 > 40), # T2
        (pct_c2h4 > 50),                                       # T3
        (pct_h2 > 40) & (pct_h2 <= 98) & (pct_ch4 < 40),     # D1
        (pct_h2 > 20) & (pct_h2 <= 40),                       # D2
    ]
    choices = ["PD", "T1", "T2", "T3", "D1", "D2"]
    df["duval_zone"] = np.select(conditions, choices, default="DT")

    # Encode numérique pour les modèles ML
    zone_map = {"PD": 0, "T1": 1, "T2": 2, "T3": 3, "D1": 4, "D2": 5, "DT": 6}
    df["duval_zone_code"] = df["duval_zone"].map(zone_map).fillna(6).astype(int)

    logger.debug("Triangle de Duval calculé.")
    return df


def add_rogers_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratios de Rogers (IEC 60599) pour le diagnostic de type de défaut.

    Ratios utilisés :
    R1 = CH4 / H2
    R2 = C2H4 / C2H2  (approx : C2H4 / (CH4 * 0.1 + 1) )
    R3 = C2H4 / CH4
    R4 = CO / CO2      (CO2 non disponible → remplacé par CO / CH4)

    Code de diagnostic Rogers :
    0 = Normal
    1 = Décharge partielle
    2 = Surchauffe légère
    3 = Surchauffe grave
    4 = Arcing (arc électrique)
    """
    df = df.copy()
    eps = 1e-9

    df["rogers_R1"] = df["CH4"]  / (df["H2"]  + eps)
    df["rogers_R3"] = df["C2H4"] / (df["CH4"] + eps)
    # R4 approximé sans CO2
    df["rogers_R4"] = df["CO"]   / (df["CH4"] + eps)

    # Classification simplifiée
    r1 = df["rogers_R1"]
    r3 = df["rogers_R3"]

    conditions = [
        (r1 < 0.1),                          # Décharge partielle
        (r1 >= 0.1) & (r1 < 1) & (r3 < 1),  # Surchauffe légère / normal
        (r1 >= 1)   & (r3 < 1),              # Surchauffe grave
        (r3 >= 1)   & (r3 < 3),              # Surchauffe très grave
        (r3 >= 3),                            # Arcing
    ]
    choices = [1, 0, 2, 3, 4]
    df["rogers_code"] = np.select(conditions, choices, default=0).astype(int)

    logger.debug("Ratios de Rogers calculés.")
    return df


def add_thermal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicateurs thermiques dérivés (charge normalisée, gradient thermique,
    exposition à la surcharge cumulée).
    """
    df = df.copy()

    # Charge normalisée (supposé rated = 1000 kVA par défaut)
    if "load_kva" in df.columns:
        df["load_ratio"] = df["load_kva"] / 1000.0

    # Gradient huile–ambiant
    if "top_oil_temp" in df.columns and "ambient_temp" in df.columns:
        df["temp_gradient"] = df["top_oil_temp"] - df["ambient_temp"]

    # Gradient point chaud–huile
    if "hot_spot_temp" in df.columns and "top_oil_temp" in df.columns:
        df["hot_spot_gradient"] = df["hot_spot_temp"] - df["top_oil_temp"]

    # Exposition cumulative à la surcharge (load_ratio > 1)
    if "load_ratio" in df.columns:
        overload = (df["load_ratio"] - 1.0).clip(lower=0)
        df["cumulative_overload"] = overload.cumsum()

    # Facteur d'accélération du vieillissement (Montsinger, IEC 60076-7)
    # FAV = exp((hot_spot - 98) / 6)  avec température de référence 98 °C
    if "hot_spot_temp" in df.columns:
        df["aging_factor"] = np.exp((df["hot_spot_temp"] - 98) / 6).clip(upper=50)
        df["cumulative_aging"] = df["aging_factor"].cumsum()

    logger.debug("Features thermiques ajoutées.")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodage cyclique des variables temporelles (heure, jour, mois)."""
    df = df.copy()
    idx = df.index

    if not isinstance(idx, pd.DatetimeIndex):
        logger.warning("L'index n'est pas un DatetimeIndex — features temporelles ignorées.")
        return df

    df["hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    logger.debug("Features temporelles encodées.")
    return df


def add_gas_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Total des gaz combustibles (TDCG) et ratios inter-gaz."""
    df = df.copy()
    gas_available = [c for c in GAS_COLS if c in df.columns]
    if not gas_available:
        return df

    df["TDCG"] = df[gas_available].sum(axis=1)

    eps = 1e-9
    if "H2" in df.columns and "CH4" in df.columns:
        df["H2_CH4_ratio"] = df["H2"] / (df["CH4"] + eps)
    if "CO" in df.columns and "H2" in df.columns:
        df["CO_H2_ratio"]  = df["CO"] / (df["H2"] + eps)

    logger.debug("Totaux de gaz (TDCG) ajoutés.")
    return df


# ---------------------------------------------------------------------------
# Pipeline complète de feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique toutes les transformations de feature engineering dans l'ordre.
    Retourne un DataFrame enrichi prêt pour l'entraînement.
    """
    logger.info("Début du feature engineering sur %d lignes…", len(df))

    df = add_rolling_features(df)
    df = add_rate_of_change(df)
    df = add_duval_triangle(df)
    df = add_rogers_ratios(df)
    df = add_thermal_features(df)
    df = add_time_features(df)
    df = add_gas_totals(df)

    # Suppression des colonnes inutiles pour le ML
    cols_to_drop = ["fault_type", "degradation_factor", "duval_zone"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Remplacement des NaN restants (dus aux fenêtres roulantes en début de série)
    df = df.ffill().bfill()

    n_features = df.shape[1]
    logger.info("Feature engineering terminé : %d features totales.", n_features)
    return df


def get_feature_columns(df: pd.DataFrame, exclude_targets: bool = True) -> list[str]:
    """
    Retourne la liste des colonnes de features (hors colonnes cibles / labels).
    """
    target_cols = {"health_index", "failure_label"}
    raw_cols    = set(["load_kva", "ambient_temp", "top_oil_temp", "hot_spot_temp",
                       "H2", "CO", "C2H4", "CH4", "moisture_ppm", "vibration_mms"])

    cols = [c for c in df.columns if c not in target_cols]
    if exclude_targets:
        return cols
    return list(df.columns)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_and_prepare

    df = load_and_prepare("data/raw/transformer_data.csv")
    df_feat = build_features(df)
    print(f"Colonnes après feature engineering ({df_feat.shape[1]}) :")
    print(df_feat.columns.tolist())
    print(df_feat.head(3))
