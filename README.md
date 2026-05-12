# PredictiveTransfo ⚡

> **Solution complète de maintenance prédictive pour transformateurs de puissance HTA/BT**
> Pipeline ML de bout en bout • Dashboard Streamlit interactif • Déploiement Docker

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![IEC 60076-7](https://img.shields.io/badge/Standard-IEC%2060076--7-green.svg)]()
[![IEC 60599](https://img.shields.io/badge/Standard-IEC%2060599-green.svg)]()

---

## Table des Matières

1. [Présentation](#1-présentation)
2. [Architecture de la Pipeline](#2-architecture-de-la-pipeline)
3. [Structure du Projet](#3-structure-du-projet)
4. [Installation](#4-installation)
5. [Utilisation](#5-utilisation)
6. [Modules Détaillés](#6-modules-détaillés)
7. [Modèles ML](#7-modèles-ml)
8. [Dashboard Streamlit](#8-dashboard-streamlit)
9. [Déploiement Docker](#9-déploiement-docker)
10. [Standards et Références](#10-standards-et-références)
11. [FAQ](#11-faq)

---

## 1. Présentation

**PredictiveTransfo** est une solution industrielle de maintenance prédictive dédiée aux **transformateurs de puissance HTA/BT** (Haute Tension A / Basse Tension). Elle exploite les données de surveillance en continu pour :

- **Anticiper** les défaillances jusqu'à **30 jours à l'avance**
- **Classifier** les types de défauts (thermique, électrique, humidité) selon les normes IEC
- **Prévoir** l'évolution des paramètres critiques (température d'huile, gaz dissous DGA)
- **Alerter** les équipes de maintenance via un tableau de bord interactif

### Paramètres surveillés

| Paramètre | Unité | Norme de référence |
|-----------|-------|-------------------|
| Charge | kVA | — |
| Température de l'huile (Top Oil) | °C | IEC 60076-7 |
| Température du point chaud (Hot Spot) | °C | IEC 60076-7 |
| Hydrogène (H₂) | ppm | IEC 60599 |
| Monoxyde de carbone (CO) | ppm | IEC 60599 |
| Éthylène (C₂H₄) | ppm | IEC 60599 |
| Méthane (CH₄) | ppm | IEC 60599 |
| Humidité dans l'huile | ppm | IEC 60422 |
| Vibrations acoustiques | mm/s | — |

---

## 2. Architecture de la Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     DONNÉES BRUTES (CSV / Capteurs)             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA GENERATOR          src/data_generator.py               │
│     • Simulation 2 ans de données horaires (17 520 points)      │
│     • Modèle thermique IEC 60076-7                              │
│     • DGA par dégradation progressive                           │
│     • Injection d'anomalies (décharges partielles, surchauffe) │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. DATA LOADER              src/data_loader.py                  │
│     • Détection et validation des colonnes                      │
│     • Correction des valeurs hors plages physiques              │
│     • Rééchantillonnage et interpolation                        │
│     • Split temporel train/test (pas de data leakage)          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. FEATURE ENGINEERING      src/features.py                     │
│     • Moyennes mobiles 7j / 30j + écarts-types                 │
│     • Taux de variation (dérivées 1h, 24h, 7j)                 │
│     • Triangle de Duval (zones PD, T1-T3, D1-D2)              │
│     • Ratios de Rogers (diagnostic de défaut)                  │
│     • Indicateurs thermiques (facteur de vieillissement)        │
│     • TDCG (Total Dissolved Combustible Gas) IEEE C57.104      │
│     • Encodage cyclique du temps (sin/cos)                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
┌────────────────────────┐  ┌────────────────────────────────────┐
│  4a. FORECASTING        │  │  4b. CLASSIFICATION                │
│  src/train.py           │  │  src/train.py                      │
│                         │  │                                    │
│  XGBoost Regressor par  │  │  • Random Forest Classifier        │
│  paramètre :            │  │  • Gradient Boosting               │
│  • top_oil_temp         │  │  • XGBoost Classifier (meilleur)  │
│  • hot_spot_temp        │  │                                    │
│  • H2, CO, C2H4, CH4   │  │  Cible : failure_label             │
│  • moisture_ppm         │  │  (défaillance dans 30 jours)      │
│                         │  │                                    │
│  Horizons : j+7, j+30  │  │  CV temporelle (5 folds)          │
│  CV temporelle 5 folds  │  │  Métriques : F1, AUC-ROC          │
└────────────┬───────────┘  └───────────────┬────────────────────┘
             │                              │
             └──────────────┬───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. INFÉRENCE               src/inference.py                    │
│     • Chargement lazy des modèles (singleton)                   │
│     • Prédiction indice de santé (0-100 %)                     │
│     • Prédiction probabilité de défaillance                     │
│     • Génération d'alertes par seuil                           │
│     • API predict_single() pour un relevé unique               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. TABLEAU DE BORD          app.py (Streamlit)                 │
│     • Vue d'ensemble + jauge de santé                          │
│     • Graphiques paramètres réels vs prédits                   │
│     • Analyse DGA (Triangle de Duval, TDCG)                   │
│     • Centre d'alertes de maintenance                          │
│     • Upload CSV → prédiction en direct                        │
│     • Métriques de performance des modèles                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Structure du Projet

```
predictive-transfo/
│
├── app.py                      # Dashboard Streamlit (point d'entrée UI)
├── requirements.txt            # Dépendances Python
├── Dockerfile                  # Conteneurisation multi-étapes
├── .dockerignore
├── README.md
│
├── data/
│   ├── raw/
│   │   └── transformer_data.csv    # Dataset généré (17 520 lignes)
│   └── processed/                  # Réservé aux données transformées
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Génération de données synthétiques réalistes
│   ├── data_loader.py          # Chargement, validation, nettoyage
│   ├── features.py             # Feature engineering (Duval, Rogers, etc.)
│   ├── train.py                # Entraînement des modèles ML
│   └── inference.py            # Inférence et génération d'alertes
│
└── models/
    ├── classifier_xgb.pkl      # Meilleur classifieur de défaillance
    ├── classifier_rf.pkl       # Random Forest (comparaison)
    ├── classifier_gb.pkl       # Gradient Boosting (comparaison)
    ├── scaler_classifier.pkl   # StandardScaler du classifieur
    ├── feature_cols.pkl        # Liste des features utilisées
    ├── forecast_<param>_<hor>.pkl  # Modèles de forecasting
    ├── scaler_forecast_*.pkl   # Scalers des forecasters
    ├── classifier_metrics.json # Métriques de performance
    └── forecast_metrics.json   # Métriques RMSE/MAE
```

---

## 4. Installation

### Prérequis

- Python 3.10 ou 3.11
- pip >= 23
- (Optionnel) Docker >= 24

### Installation locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/azzedbenj007/predictive-transfo.git
cd predictive-transfo

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Utilisation

### Étape 1 — Générer les données de démonstration

```bash
python -m src.data_generator
```

Génère `data/raw/transformer_data.csv` avec **17 520 relevés horaires** sur 2 ans,
incluant des anomalies injectées aléatoirement.

### Étape 2 — Entraîner les modèles

```bash
python -m src.train
```

Durée estimée : **3-8 minutes** selon le CPU. Les modèles sont sauvegardés dans `models/`.

Sortie typique :
```
2024-01-01 10:00:00 [INFO] === Entraînement des modèles de forecasting ===
2024-01-01 10:00:05 [INFO]   → top_oil_temp | horizon 7d
2024-01-01 10:00:10 [INFO]     RMSE CV = 2.341 ± 0.218 | MAE CV = 1.876
...
2024-01-01 10:05:30 [INFO] XGB → F1 CV = 0.891 | AUC CV = 0.967
```

### Étape 3 — Lancer le dashboard

```bash
streamlit run app.py
```

Ouvrir le navigateur sur **http://localhost:8501**

### Commandes en une ligne (développement rapide)

```bash
python -m src.data_generator && python -m src.train && streamlit run app.py
```

---

## 6. Modules Détaillés

### `src/data_generator.py` — Génération de données

**Classe :** `TransformerDataGenerator`

Simule un transformateur 20kV/400V, 1 MVA, avec les modèles physiques suivants :

| Modèle | Description |
|--------|-------------|
| **Charge** | Profil journalier + hebdomadaire + saisonnalité été/hiver |
| **Température huile** | Modèle IEC 60076-7 avec constante de temps thermique |
| **Point chaud** | Gradient bobine selon IEC 60076-7 (rapport de charge^1.3) |
| **DGA** | Dépendance thermique + dégradation progressive |
| **Indice de santé** | Pondération multi-paramètres (0-100 %) |

**Types d'anomalies injectées :**
- `partial_discharge` — Pic H2 x 3-8, C2H4 x 2-5
- `overheating` — Température +20-40 °C, CH4 x 2-4
- `moisture_fault` — Humidité x 3-6

### `src/data_loader.py` — Chargement des données

**Classe :** `DataLoader`

```python
from src.data_loader import load_and_prepare

df = load_and_prepare("data/raw/transformer_data.csv")
# Retourne un DataFrame propre avec index DatetimeIndex
```

**Fonctionnalités :**
- Détection automatique de la colonne timestamp
- Validation des plages physiques (ex : T_huile dans [0, 150] °C)
- Rééchantillonnage à la fréquence cible (défaut : 1h)
- Interpolation temporelle des NaN
- Split train/test **temporel** (pas de data leakage)

### `src/features.py` — Feature Engineering

**Fonction principale :** `build_features(df)`

| Feature | Description | Référence |
|---------|-------------|-----------|
| `*_mean_168h` | Moyenne mobile 7 jours | — |
| `*_mean_720h` | Moyenne mobile 30 jours | — |
| `*_roc_1h/24h/168h` | Taux de variation | — |
| `duval_pct_CH4/C2H4/H2` | Coordonnées du triangle | IEC 60599 |
| `duval_zone` | Zone de défaut (PD, T1-T3, D1-D2, DT) | IEC 60599 |
| `rogers_R1/R3/R4` | Ratios de Rogers | IEC 60599 |
| `rogers_code` | Code de diagnostic (0-4) | IEC 60599 |
| `aging_factor` | Facteur d'accélération Montsinger | IEC 60076-7 |
| `cumulative_aging` | Vieillissement cumulé | IEC 60076-7 |
| `TDCG` | Total gaz combustibles dissous | IEEE C57.104 |
| `hour_sin/cos` | Encodage cyclique heure | — |
| `month_sin/cos` | Encodage cyclique mois | — |

### `src/train.py` — Entraînement

**Validation croisée temporelle :**

```
Fold 1 : [========-------------------------------------]
Fold 2 : [=============--------------------------------]
Fold 3 : [====================--------------------------]
Fold 4 : [============================-----------------]
Fold 5 : [====================================----------]
         |________ Train ________|_____ Test _____|
```

Garantit qu'aucune donnée future ne contamine l'entraînement.

**Modèles :**

| Tâche | Modèle | Hyperparamètres |
|-------|--------|-----------------|
| Forecasting | XGBRegressor | n_est=400, depth=6, lr=0.05 |
| Classification | XGBClassifier | n_est=400, depth=6, lr=0.05 |
| Classification | RandomForestClassifier | n_est=300, depth=12 |
| Classification | GradientBoostingClassifier | n_est=300, depth=5, lr=0.05 |

### `src/inference.py` — Inférence

**Classe :** `TransformerPredictor`

```python
from src.inference import get_predictor

predictor = get_predictor()  # Singleton — chargement unique

# Prédiction sur un DataFrame historique
df_feat = build_features(df)
result = predictor.predict_health(df_feat)
# Colonnes ajoutées : failure_proba, health_score, risk_level

# Prédiction d'un paramètre futur
preds_7d = predictor.predict_forecast(df_feat, "top_oil_temp", horizon="7d")

# Prédiction sur un seul relevé
row = {"load_kva": 750, "top_oil_temp": 72, "H2": 45}
result = predictor.predict_single(row)
# {'failure_proba': 0.12, 'health_score': 88.0, 'risk_level': 'NORMAL', 'alerts': []}
```

**Seuils d'alerte (configurables dans `inference.py`) :**

| Paramètre | Seuil ALERTE | Seuil CRITIQUE (x1.3) |
|-----------|-------------|----------------------|
| Température huile | 100 °C | 130 °C |
| Point chaud | 130 °C | 169 °C |
| H2 | 150 ppm | 195 ppm |
| C2H4 | 50 ppm | 65 ppm |
| CH4 | 120 ppm | 156 ppm |
| CO | 500 ppm | 650 ppm |
| Humidité | 30 ppm | 39 ppm |
| Vibrations | 5 mm/s | 6.5 mm/s |

---

## 7. Modèles ML

### Forecasting (XGBoost Regressor)

**Features d'entrée :**
- Lags 1h-24h de la variable cible
- Toutes les features ingénierées actuelles

**Performances typiques (CV temporelle 5 folds) :**

| Paramètre | Horizon | RMSE CV | MAE CV |
|-----------|---------|---------|--------|
| top_oil_temp | j+7 | ~2.5 °C | ~1.9 °C |
| top_oil_temp | j+30 | ~4.1 °C | ~3.2 °C |
| H2 | j+7 | ~8 ppm | ~6 ppm |
| H2 | j+30 | ~14 ppm | ~11 ppm |

### Classifieur de Défaillance (XGBoost Classifier)

**Cible :** `failure_label` = 1 si l'indice de santé descend < 50 % dans les 30 jours

**Performances typiques :**

| Modèle | F1-Score CV | AUC-ROC CV |
|--------|-------------|-----------|
| XGBoost | ~0.89 | ~0.97 |
| Random Forest | ~0.85 | ~0.94 |
| Gradient Boosting | ~0.87 | ~0.95 |

**Indice de santé :**

```
health_score = (1 - failure_proba) x 100

Niveaux :
  0-40 %   → CRITIQUE  (intervention immédiate)
  40-65 %  → ALERTE    (planifier une inspection)
  65-100 % → NORMAL    (bon état de fonctionnement)
```

---

## 8. Dashboard Streamlit

L'interface est composée de **6 sections** accessibles depuis la barre latérale :

### Vue d'ensemble
- 5 KPIs : Indice de santé, Prob. Défaillance, T° Huile, H2, Vibrations
- Jauge animée de l'indice de santé avec zones colorées
- Graphique combiné Charge + Températures
- Alertes actives en temps réel

### Paramètres & Prévisions
- Sélecteur de paramètre parmi les 8 variables surveillées
- Graphique valeurs réelles vs prédites (horizon configurable : j+7 ou j+30)
- Ligne de seuil d'alarme superposée
- Matrice de corrélation interactive

### Analyse DGA
- Évolution temporelle des 4 gaz dissous (H2, CO, C2H4, CH4)
- Camembert des zones de Duval (distribution des types de défauts)
- Scatter Triangle de Duval (positionnement CH4 vs C2H4)
- TDCG avec niveaux IEEE C57.104 annotés

### Alertes
- Alertes sur le dernier relevé (CRITIQUE / ALERTE / NORMAL)
- Historique des dépassements de seuils (tableau téléchargeable)
- Histogramme de fréquence des dépassements par paramètre
- Distribution temporelle du niveau de risque

### Prédiction CSV
- Upload d'un fichier CSV de relevés personnalisés
- Prédiction automatique et affichage des KPIs
- Graphique de l'indice de santé sur le fichier uploadé
- Téléchargement des prédictions au format CSV
- Template CSV téléchargeable

### Métriques Modèles
- Tableau des métriques RMSE/MAE par paramètre et horizon
- Rapport de classification complet (precision, recall, F1)
- Matrice de confusion interactive
- Inventaire des modèles sauvegardés avec tailles

---

## 9. Déploiement Docker

### Construction de l'image

```bash
docker build -t predictive-transfo:latest .
```

### Lancement du conteneur

```bash
# Mode standard
docker run -p 8501:8501 predictive-transfo:latest

# Avec volumes persistants (données et modèles)
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  predictive-transfo:latest
```

Accès : **http://localhost:8501**

### Docker Compose (recommandé)

```yaml
version: "3.9"
services:
  predictive-transfo:
    image: predictive-transfo:latest
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker compose up -d
docker compose logs -f
```

### Variables d'environnement

| Variable | Valeur par défaut | Description |
|----------|------------------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Port d'écoute |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Adresse d'écoute |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Mode sans navigateur |

---

## 10. Standards et Références

| Norme | Application |
|-------|-------------|
| **IEC 60076-7** | Modèle thermique huile/point chaud, facteur de vieillissement Montsinger |
| **IEC 60599** | Interprétation DGA, Triangle de Duval, Ratios de Rogers |
| **IEC 60422** | Surveillance de l'humidité dans l'huile isolante |
| **IEEE C57.104** | Niveaux TDCG (Total Dissolved Combustible Gas) |
| **IEEE C57.91** | Guide de chargement des transformateurs à huile |

### Triangle de Duval — Zones de défauts

```
Zones :
  PD  = Partial Discharge (décharges partielles)
  T1  = Défaut thermique < 300 °C
  T2  = Défaut thermique 300-700 °C
  T3  = Défaut thermique > 700 °C
  D1  = Décharge électrique de faible énergie
  D2  = Décharge électrique de haute énergie
  DT  = Défaut mixte thermique + électrique
```

---

## 11. FAQ

**Q : Combien de temps dure l'entraînement ?**
R : Entre 3 et 8 minutes sur un CPU standard (Intel i5/i7), selon le nombre de features. XGBoost avec `n_jobs=-1` parallélise les calculs.

**Q : Puis-je utiliser mes propres données de terrain ?**
R : Oui. Préparez un CSV avec les colonnes listées dans la section "Format CSV attendu" du dashboard (onglet Prédiction CSV). Uploadez-le directement dans l'interface.

**Q : Les seuils d'alerte sont-ils modifiables ?**
R : Oui, dans `src/inference.py`, le dictionnaire `ALERT_THRESHOLDS` contient tous les seuils. Modifiez-les selon les spécifications du fabricant.

**Q : Comment adapter le modèle à un transformateur de capacité différente ?**
R : Modifiez le paramètre `rated_kva` dans `TransformerDataGenerator` et régénérez les données, puis ré-entraînez les modèles.

**Q : Est-ce que les prédictions fonctionnent sans modèles entraînés ?**
R : Le dashboard affiche les données historiques et les alertes basées sur les seuils sans modèles. Les prédictions ML nécessitent un entraînement préalable.

**Q : Comment surveiller plusieurs transformateurs ?**
R : Instanciez plusieurs `TransformerPredictor` avec différents répertoires de modèles, ou ajoutez un identifiant de transformateur (`transformer_id`) dans le dataset.

---

## Licence

MIT License — PredictiveTransfo

---

*Solution conforme aux standards IEC 60076-7, IEC 60599 et IEEE C57.104 pour la maintenance prédictive des transformateurs de puissance.*
