"""
Chargement, validation et préparation des données brutes du transformateur.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Colonnes attendues dans le CSV brut
REQUIRED_COLUMNS = [
    "load_kva",
    "ambient_temp",
    "top_oil_temp",
    "hot_spot_temp",
    "H2",
    "CO",
    "C2H4",
    "CH4",
    "moisture_ppm",
    "vibration_mms",
]

# Plages physiques acceptables (min, max) par colonne
PHYSICAL_BOUNDS = {
    "load_kva":       (0,    1500),
    "ambient_temp":   (-20,  50),
    "top_oil_temp":   (0,    150),
    "hot_spot_temp":  (0,    200),
    "H2":             (0,    2000),
    "CO":             (0,    5000),
    "C2H4":           (0,    1000),
    "CH4":            (0,    2000),
    "moisture_ppm":   (0,    100),
    "vibration_mms":  (0,    20),
}


class DataLoader:
    """
    Charge et nettoie un fichier CSV de données de transformateur.
    Gère la validation, le rééchantillonnage et le remplissage des NaN.
    """

    def __init__(self, filepath: str, freq: str = "1h"):
        self.filepath = Path(filepath)
        self.freq = freq
        self._df_raw: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    def load(self) -> "DataLoader":
        """Lit le CSV et détecte la colonne index temporelle."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.filepath}")

        logger.info("Chargement de %s …", self.filepath)
        try:
            df = pd.read_csv(self.filepath, parse_dates=True)
        except Exception as exc:
            raise ValueError(f"Erreur de lecture CSV : {exc}") from exc

        # Détection de la colonne timestamp
        df = self._detect_and_set_index(df)
        self._df_raw = df
        logger.info("Données brutes chargées : %d lignes, %d colonnes.", *df.shape)
        return self

    def _detect_and_set_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tente de trouver et d'utiliser une colonne datetime comme index."""
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        candidate_names = ["timestamp", "date", "datetime", "time", "index"]
        for col in df.columns:
            if col.lower() in candidate_names:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    df.index.name = "timestamp"
                    return df
                except Exception:
                    continue

        # Dernier recours : essayer la première colonne
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
            df.index.name = "timestamp"
            return df
        except Exception as exc:
            raise ValueError(
                "Impossible de détecter une colonne datetime dans le CSV. "
                "Assurez-vous qu'une colonne 'timestamp' existe."
            ) from exc

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> "DataLoader":
        """Vérifie la présence des colonnes obligatoires et les plages physiques."""
        if self._df_raw is None:
            raise RuntimeError("Appelez d'abord .load().")

        df = self._df_raw
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        violations = {}
        for col, (lo, hi) in PHYSICAL_BOUNDS.items():
            if col not in df.columns:
                continue
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_out > 0:
                violations[col] = n_out

        if violations:
            logger.warning(
                "Valeurs hors plage physique détectées : %s — elles seront remplacées par NaN.",
                violations,
            )
            for col, (lo, hi) in PHYSICAL_BOUNDS.items():
                if col in df.columns:
                    df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
            self._df_raw = df

        logger.info("Validation terminée.")
        return self

    # ------------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------------
    def clean(self) -> "DataLoader":
        """
        - Rééchantillonne à la fréquence cible
        - Trie l'index
        - Interpole les NaN (méthode time) puis forward-fill pour les extrémités
        """
        if self._df_raw is None:
            raise RuntimeError("Appelez d'abord .load().")

        df = self._df_raw.sort_index()

        # Suppression des doublons d'index (garder premier)
        df = df[~df.index.duplicated(keep="first")]

        # Rééchantillonnage
        numeric_cols = df.select_dtypes(include="number").columns
        cat_cols     = df.select_dtypes(exclude="number").columns

        df_num = df[numeric_cols].resample(self.freq).mean()

        if len(cat_cols) > 0:
            df_cat = df[cat_cols].resample(self.freq).last()
            df = pd.concat([df_num, df_cat], axis=1)
        else:
            df = df_num

        # Interpolation temporelle des trous
        df[numeric_cols] = (
            df[numeric_cols]
            .interpolate(method="time")
            .ffill()
            .bfill()
        )

        # Remplissage des colonnes catégorielles
        if len(cat_cols) > 0:
            df[cat_cols] = df[cat_cols].ffill().bfill().fillna("normal")

        n_nan = df.isnull().sum().sum()
        if n_nan > 0:
            logger.warning("%d valeurs NaN résiduelles après nettoyage.", n_nan)

        self._df_raw = df
        logger.info("Nettoyage terminé : %d lignes conservées.", len(df))
        return self

    # ------------------------------------------------------------------
    # Accesseur
    # ------------------------------------------------------------------
    def get_dataframe(self) -> pd.DataFrame:
        """Retourne le DataFrame nettoyé."""
        if self._df_raw is None:
            raise RuntimeError("Appelez .load().validate().clean() avant get_dataframe().")
        return self._df_raw.copy()

    # ------------------------------------------------------------------
    # Séparation train / test temporelle
    # ------------------------------------------------------------------
    @staticmethod
    def train_test_split(
        df: pd.DataFrame, test_ratio: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Découpe temporelle stricte : les données de test sont toujours postérieures
        aux données d'entraînement (pas de mélange aléatoire).
        """
        split_idx = int(len(df) * (1 - test_ratio))
        train = df.iloc[:split_idx].copy()
        test  = df.iloc[split_idx:].copy()
        logger.info(
            "Split train/test : %d / %d lignes (ratio test=%.0f%%).",
            len(train), len(test), test_ratio * 100,
        )
        return train, test


# ---------------------------------------------------------------------------
# Utilitaire standalone
# ---------------------------------------------------------------------------
def load_and_prepare(filepath: str, freq: str = "1h") -> pd.DataFrame:
    """Raccourci : charge, valide, nettoie et retourne un DataFrame prêt."""
    loader = DataLoader(filepath, freq=freq)
    return loader.load().validate().clean().get_dataframe()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_and_prepare("data/raw/transformer_data.csv")
    print(df.head())
    print(df.dtypes)
