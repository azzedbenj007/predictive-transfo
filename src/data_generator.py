"""
Générateur de données synthétiques réalistes pour un transformateur de puissance HTA/BT.
Simule 2 ans de mesures horaires avec tendances d'usure, cycles saisonniers et anomalies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TransformerDataGenerator:
    """
    Génère un dataset temporel simulant un transformateur de puissance 20kV/400V, 1 MVA.
    Inclut usure progressive, saisonnalité, bruit et injections d'anomalies.
    """

    def __init__(self, seed: int = 42, rated_kva: float = 1000.0):
        np.random.seed(seed)
        random.seed(seed)
        self.rated_kva = rated_kva
        self.anomaly_log = []

    # ------------------------------------------------------------------
    # Profils de charge
    # ------------------------------------------------------------------
    def _load_profile(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Charge en kVA avec cycle journalier, hebdomadaire et saisonnier."""
        n = len(timestamps)
        hour = timestamps.hour.values
        dayofweek = timestamps.dayofweek.values  # 0=lundi … 6=dimanche
        dayofyear = timestamps.dayofyear.values

        # Profil horaire (courbe typique réseau BT)
        hourly = (
            0.4
            + 0.3 * np.sin(np.pi * (hour - 6) / 12) ** 2
            + 0.15 * np.sin(2 * np.pi * hour / 24)
        )
        hourly = np.clip(hourly, 0.2, 1.0)

        # Réduction week-end
        weekend_mask = (dayofweek >= 5).astype(float)
        hourly *= 1 - 0.2 * weekend_mask

        # Saisonnalité : pic été (climatisation) et pic hiver (chauffage)
        seasonal = 1.0 + 0.25 * np.cos(2 * np.pi * (dayofyear - 15) / 365)  # pic hiver
        seasonal += 0.15 * np.cos(2 * np.pi * (dayofyear - 200) / 365)       # pic été

        # Tendance d'usure légère sur 2 ans (charge moyenne +3 %)
        trend = 1.0 + 0.03 * np.linspace(0, 1, n)

        load = hourly * seasonal * trend * self.rated_kva
        noise = np.random.normal(0, 20, n)
        return np.clip(load + noise, 50, self.rated_kva * 1.15)

    # ------------------------------------------------------------------
    # Température huile
    # ------------------------------------------------------------------
    def _top_oil_temperature(
        self, load_kva: np.ndarray, ambient_temp: np.ndarray
    ) -> np.ndarray:
        """Température huile en °C selon IEC 60076-7 (modèle simplifié)."""
        load_ratio = load_kva / self.rated_kva
        # Élévation à pleine charge ≈ 55 °C
        delta_oil = 55 * (load_ratio ** 1.6)
        # Inertie thermique (filtre exponentiel)
        top_oil = np.zeros(len(load_kva))
        top_oil[0] = ambient_temp[0] + delta_oil[0]
        tau = 0.85  # constante de temps thermique horaire
        for i in range(1, len(load_kva)):
            top_oil[i] = tau * top_oil[i - 1] + (1 - tau) * (
                ambient_temp[i] + delta_oil[i]
            )
        noise = np.random.normal(0, 1.5, len(load_kva))
        return top_oil + noise

    # ------------------------------------------------------------------
    # Température point chaud
    # ------------------------------------------------------------------
    def _hot_spot_temperature(
        self, top_oil: np.ndarray, load_kva: np.ndarray
    ) -> np.ndarray:
        """Point chaud = huile + gradient bobine (IEC 60076-7)."""
        load_ratio = load_kva / self.rated_kva
        # Gradient à pleine charge ≈ 23 °C
        gradient = 23 * (load_ratio ** 1.3)
        noise = np.random.normal(0, 2.0, len(top_oil))
        return top_oil + gradient + noise

    # ------------------------------------------------------------------
    # Températures ambiantes
    # ------------------------------------------------------------------
    def _ambient_temperature(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Température ambiante simulée (climat tempéré, France)."""
        doy = timestamps.dayofyear.values
        hour = timestamps.hour.values
        seasonal = 12 + 12 * np.sin(2 * np.pi * (doy - 80) / 365)  # 0–24 °C
        diurnal = 3 * np.sin(2 * np.pi * (hour - 6) / 24)
        noise = np.random.normal(0, 2, len(timestamps))
        return seasonal + diurnal + noise

    # ------------------------------------------------------------------
    # Gaz dissous (DGA)
    # ------------------------------------------------------------------
    def _dga_gases(
        self,
        n: int,
        top_oil: np.ndarray,
        hot_spot: np.ndarray,
        degradation_factor: np.ndarray,
    ) -> dict:
        """
        Gaz dissous en ppm avec tendance d'usure thermique.
        Modèle inspiré IEC 60599 (niveaux typiques transformateur sain).
        """
        # Tendance lente liée à la dégradation thermique
        base_h2  = 20  + 30  * degradation_factor + 8  * (hot_spot / 120) ** 2
        base_co  = 200 + 100 * degradation_factor + 20 * (top_oil / 80)
        base_c2h4 = 5  + 15  * degradation_factor + 10 * (hot_spot / 120) ** 3
        base_ch4  = 30 + 40  * degradation_factor + 15 * (hot_spot / 100)

        gases = {
            "H2":   np.clip(base_h2  + np.random.normal(0, 3,  n), 0, None),
            "CO":   np.clip(base_co  + np.random.normal(0, 15, n), 0, None),
            "C2H4": np.clip(base_c2h4 + np.random.normal(0, 2, n), 0, None),
            "CH4":  np.clip(base_ch4 + np.random.normal(0, 5,  n), 0, None),
        }
        return gases

    # ------------------------------------------------------------------
    # Humidité dans l'huile
    # ------------------------------------------------------------------
    def _moisture_in_oil(
        self, n: int, top_oil: np.ndarray, degradation_factor: np.ndarray
    ) -> np.ndarray:
        """Teneur en eau (ppm) – augmente avec dégradation et température."""
        base = 8 + 12 * degradation_factor + 0.05 * top_oil
        noise = np.random.normal(0, 1.5, n)
        return np.clip(base + noise, 0.5, 60)

    # ------------------------------------------------------------------
    # Vibrations acoustiques
    # ------------------------------------------------------------------
    def _vibrations(
        self, load_kva: np.ndarray, degradation_factor: np.ndarray
    ) -> np.ndarray:
        """Amplitude vibratoire en mm/s – liée à la charge et à l'usure."""
        load_ratio = load_kva / self.rated_kva
        vib = 1.5 + 1.5 * load_ratio + 2.5 * degradation_factor
        noise = np.random.normal(0, 0.3, len(load_kva))
        return np.clip(vib + noise, 0, None)

    # ------------------------------------------------------------------
    # Injections d'anomalies
    # ------------------------------------------------------------------
    def _inject_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Injecte aléatoirement des événements de défaillance imminente :
        - Décharges partielles : pic H2 et C2H4
        - Surchauffe ponctuelle : pic température + gaz thermiques
        - Défaut humidité : pic moisture
        """
        df = df.copy()
        n = len(df)
        df["fault_type"] = "normal"
        df["health_index"] = 100.0  # sera recalculé

        # Nombre d'anomalies par type (aléatoire)
        rng = np.random.default_rng(99)
        n_partial_discharge = rng.integers(3, 8)
        n_overheating       = rng.integers(4, 10)
        n_moisture_fault    = rng.integers(2, 5)

        def _random_window(duration_h: int):
            start = rng.integers(0, n - duration_h)
            return slice(start, start + duration_h)

        # --- Décharges partielles ---
        for _ in range(n_partial_discharge):
            sl = _random_window(rng.integers(48, 168))
            df.iloc[sl, df.columns.get_loc("H2")]   *= rng.uniform(3, 8)
            df.iloc[sl, df.columns.get_loc("C2H4")] *= rng.uniform(2, 5)
            df.iloc[sl, df.columns.get_loc("fault_type")] = "partial_discharge"
            self.anomaly_log.append(
                {"type": "partial_discharge", "start": df.index[sl.start], "end": df.index[sl.stop - 1]}
            )

        # --- Surchauffe ---
        for _ in range(n_overheating):
            sl = _random_window(rng.integers(24, 120))
            df.iloc[sl, df.columns.get_loc("top_oil_temp")] += rng.uniform(20, 40)
            df.iloc[sl, df.columns.get_loc("hot_spot_temp")] += rng.uniform(25, 50)
            df.iloc[sl, df.columns.get_loc("CH4")]  *= rng.uniform(2, 4)
            df.iloc[sl, df.columns.get_loc("C2H4")] *= rng.uniform(2, 6)
            df.iloc[sl, df.columns.get_loc("fault_type")] = "overheating"
            self.anomaly_log.append(
                {"type": "overheating", "start": df.index[sl.start], "end": df.index[sl.stop - 1]}
            )

        # --- Humidité excessive ---
        for _ in range(n_moisture_fault):
            sl = _random_window(rng.integers(72, 200))
            df.iloc[sl, df.columns.get_loc("moisture_ppm")] *= rng.uniform(3, 6)
            df.iloc[sl, df.columns.get_loc("fault_type")] = "moisture_fault"
            self.anomaly_log.append(
                {"type": "moisture_fault", "start": df.index[sl.start], "end": df.index[sl.stop - 1]}
            )

        return df

    # ------------------------------------------------------------------
    # Indice de santé
    # ------------------------------------------------------------------
    def _compute_health_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcul simplifié de l'indice de santé (0–100 %).
        Basé sur pondération normalisée des paramètres.
        """
        scores = pd.DataFrame(index=df.index)

        # Température huile (limite critique 105 °C)
        scores["s_top_oil"]  = np.clip(1 - (df["top_oil_temp"] - 40)  / 65,  0, 1)
        # Point chaud (limite critique 140 °C)
        scores["s_hot_spot"] = np.clip(1 - (df["hot_spot_temp"] - 50) / 90,  0, 1)
        # H2 (> 150 ppm = alarme)
        scores["s_h2"]       = np.clip(1 - df["H2"]   / 150, 0, 1)
        # C2H4 (> 50 ppm = alarme)
        scores["s_c2h4"]     = np.clip(1 - df["C2H4"] / 50,  0, 1)
        # CH4 (> 120 ppm = alarme)
        scores["s_ch4"]      = np.clip(1 - df["CH4"]  / 120, 0, 1)
        # CO (> 500 ppm = alarme)
        scores["s_co"]       = np.clip(1 - df["CO"]   / 500, 0, 1)
        # Humidité (> 30 ppm = alarme)
        scores["s_moisture"] = np.clip(1 - df["moisture_ppm"] / 30, 0, 1)
        # Vibrations (> 5 mm/s = alarme)
        scores["s_vib"]      = np.clip(1 - df["vibration_mms"] / 5, 0, 1)

        weights = {
            "s_top_oil": 0.20, "s_hot_spot": 0.20,
            "s_h2": 0.15, "s_c2h4": 0.10, "s_ch4": 0.10, "s_co": 0.05,
            "s_moisture": 0.10, "s_vib": 0.10,
        }
        hi = sum(scores[col] * w for col, w in weights.items()) * 100
        return np.clip(hi, 0, 100)

    # ------------------------------------------------------------------
    # Génération principale
    # ------------------------------------------------------------------
    def generate(
        self,
        start: str = "2022-01-01",
        end: str   = "2023-12-31",
        freq: str  = "1h",
    ) -> pd.DataFrame:
        """
        Génère le dataset complet et le retourne sous forme de DataFrame.
        """
        logger.info("Génération des données de %s à %s (fréquence=%s)…", start, end, freq)
        timestamps = pd.date_range(start=start, end=end, freq=freq)
        n = len(timestamps)

        # Facteur de dégradation progressif (0 → 1 sur 2 ans) + bruit
        degradation = np.linspace(0, 0.7, n) + np.random.normal(0, 0.02, n)
        degradation = np.clip(degradation, 0, 1)

        ambient = self._ambient_temperature(timestamps)
        load    = self._load_profile(timestamps)
        top_oil = self._top_oil_temperature(load, ambient)
        hot_spot = self._hot_spot_temperature(top_oil, load)
        gases   = self._dga_gases(n, top_oil, hot_spot, degradation)
        moisture = self._moisture_in_oil(n, top_oil, degradation)
        vibration = self._vibrations(load, degradation)

        df = pd.DataFrame(
            {
                "timestamp":       timestamps,
                "load_kva":        load,
                "ambient_temp":    ambient,
                "top_oil_temp":    top_oil,
                "hot_spot_temp":   hot_spot,
                "H2":              gases["H2"],
                "CO":              gases["CO"],
                "C2H4":            gases["C2H4"],
                "CH4":             gases["CH4"],
                "moisture_ppm":    moisture,
                "vibration_mms":   vibration,
                "degradation_factor": degradation,
            }
        )
        df = df.set_index("timestamp")

        # Injection d'anomalies
        df = self._inject_anomalies(df)

        # Indice de santé final
        df["health_index"] = self._compute_health_index(df)

        # Label binaire de défaillance (health_index < 50 dans les 30 prochains jours)
        window = 24 * 30  # 30 jours en heures
        df["failure_label"] = (
            df["health_index"]
            .rolling(window=window, min_periods=1)
            .min()
            .shift(-window)
            .fillna(method="ffill")
            < 50
        ).astype(int)

        logger.info("Dataset généré : %d lignes, %d colonnes.", *df.shape)
        logger.info("Anomalies injectées : %d événements.", len(self.anomaly_log))
        return df

    def save(self, df: pd.DataFrame, path: str = "data/raw/transformer_data.csv") -> None:
        """Sauvegarde le dataset au format CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        logger.info("Dataset sauvegardé → %s", path)

    def anomaly_report(self) -> pd.DataFrame:
        """Retourne le journal des anomalies injectées."""
        return pd.DataFrame(self.anomaly_log)


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gen = TransformerDataGenerator(seed=42, rated_kva=1000.0)
    df  = gen.generate(start="2022-01-01", end="2023-12-31", freq="1h")
    gen.save(df, path="data/raw/transformer_data.csv")

    print("\n=== Aperçu du dataset ===")
    print(df.head())
    print(f"\nFormes : {df.shape}")
    print(f"\nStatistiques :\n{df.describe().round(2)}")
    print(f"\n=== Anomalies injectées ===\n{gen.anomaly_report()}")
