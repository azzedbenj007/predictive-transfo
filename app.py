"""
Tableau de bord Streamlit pour la Maintenance Prédictive des Transformateurs HTA/BT.

Sections :
  1. Vue d'ensemble (KPIs + indice de santé)
  2. Paramètres en temps réel vs prédits
  3. Analyse DGA (Gaz Dissous)
  4. Alertes de maintenance
  5. Upload CSV pour prédiction en direct
  6. Métriques de performance des modèles
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ajouter la racine du projet au PYTHONPATH
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PredictiveTransfo — Maintenance Prédictive",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personnalisé
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px 0;
        border-left: 4px solid #4fc3f7;
    }
    .alert-critique {
        background-color: #7f0000;
        border-left: 5px solid #ff1744;
        padding: 10px 16px;
        border-radius: 6px;
        margin: 4px 0;
    }
    .alert-warning {
        background-color: #4a2800;
        border-left: 5px solid #ff9100;
        padding: 10px 16px;
        border-radius: 6px;
        margin: 4px 0;
    }
    .alert-normal {
        background-color: #003300;
        border-left: 5px solid #00e676;
        padding: 10px 16px;
        border-radius: 6px;
        margin: 4px 0;
    }
    h1 { color: #4fc3f7; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Fonctions utilitaires et cache
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Chargement des données historiques…")
def load_data(path: str = "data/raw/transformer_data.csv") -> pd.DataFrame:
    from src.data_loader import load_and_prepare
    return load_and_prepare(path)


@st.cache_data(show_spinner="Feature engineering…")
def compute_features(_df: pd.DataFrame) -> pd.DataFrame:
    from src.features import build_features
    return build_features(_df)


@st.cache_resource(show_spinner="Chargement des modèles ML…")
def load_predictor():
    from src.inference import TransformerPredictor
    try:
        p = TransformerPredictor()
        p.load_models()
        return p, None
    except Exception as exc:
        return None, str(exc)


def health_color(score: float) -> str:
    if score < 40:
        return "#ff1744"
    if score < 65:
        return "#ff9100"
    return "#00e676"


def risk_emoji(level: str) -> str:
    return {"CRITIQUE": "🔴", "ALERTE": "🟠", "NORMAL": "🟢"}.get(level, "⚪")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Power_Transformer.jpg/320px-Power_Transformer.jpg",
        use_container_width=True,
    )
    st.title("⚡ PredictiveTransfo")
    st.caption("Maintenance prédictive — Transformateur HTA/BT 1 MVA")

    st.divider()
    page = st.radio(
        "Navigation",
        [
            "📊 Vue d'ensemble",
            "📈 Paramètres & Prévisions",
            "🧪 Analyse DGA",
            "🚨 Alertes",
            "📤 Prédiction CSV",
            "📉 Métriques Modèles",
        ],
    )

    st.divider()
    st.subheader("Paramètres d'affichage")
    window_days = st.slider("Fenêtre d'affichage (jours)", 7, 180, 30)
    horizon     = st.selectbox("Horizon de prévision", ["7d", "30d"])

    # Génération / rechargement des données
    if st.button("🔄 Régénérer les données mock"):
        st.cache_data.clear()
        st.rerun()

    if st.button("🧠 Lancer l'entraînement"):
        with st.spinner("Entraînement en cours… (~5 min)"):
            try:
                from src.train import run_training_pipeline
                run_training_pipeline()
                st.cache_resource.clear()
                st.success("Entraînement terminé !")
            except Exception as exc:
                st.error(f"Erreur : {exc}")


# ---------------------------------------------------------------------------
# Chargement global des données
# ---------------------------------------------------------------------------
RAW_DATA_PATH = "data/raw/transformer_data.csv"

# Génère les données si elles n'existent pas
if not Path(RAW_DATA_PATH).exists():
    with st.spinner("Génération des données de démonstration…"):
        from src.data_generator import TransformerDataGenerator
        gen = TransformerDataGenerator(seed=42)
        df_raw = gen.generate()
        gen.save(df_raw, RAW_DATA_PATH)
        st.success("Données générées avec succès.")

try:
    df_raw  = load_data(RAW_DATA_PATH)
    df_feat = compute_features(df_raw)
except Exception as exc:
    st.error(f"Impossible de charger les données : {exc}")
    st.stop()

predictor, pred_error = load_predictor()

# Appliquer la fenêtre temporelle
cutoff = df_raw.index[-1] - pd.Timedelta(days=window_days)
df_view      = df_raw[df_raw.index >= cutoff]
df_feat_view = df_feat[df_feat.index >= cutoff]

# Prédictions (si modèles disponibles)
df_pred = None
if predictor:
    try:
        df_pred = predictor.predict_health(df_feat_view)
    except Exception as exc:
        logger.warning("Prédictions indisponibles : %s", exc)


# ===========================================================================
# PAGE 1 — Vue d'ensemble
# ===========================================================================
if page == "📊 Vue d'ensemble":
    st.title("📊 Vue d'ensemble du Transformateur")

    # --- KPIs ---
    latest = df_raw.iloc[-1]
    health_score = float(df_pred["health_score"].iloc[-1]) if df_pred is not None else None
    fail_proba   = float(df_pred["failure_proba"].iloc[-1]) if df_pred is not None else None
    risk         = str(df_pred["risk_level"].iloc[-1]) if df_pred is not None else "N/A"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        val = health_score if health_score is not None else 0
        st.metric("🏥 Indice de Santé", f"{val:.1f}%",
                  delta=None, help="0 = défaillance, 100 = état parfait")
    with col2:
        st.metric("⚠️ Prob. Défaillance", f"{fail_proba*100:.1f}%" if fail_proba is not None else "N/A",
                  help="Probabilité de défaillance dans les 30 jours")
    with col3:
        st.metric("🌡️ Température Huile", f"{latest['top_oil_temp']:.1f} °C",
                  delta=f"{latest['top_oil_temp'] - 60:.1f}°C vs nominal")
    with col4:
        st.metric("💧 H₂ (DGA)", f"{latest['H2']:.0f} ppm",
                  help="Seuil alarme : 150 ppm")
    with col5:
        st.metric("🔊 Vibrations", f"{latest['vibration_mms']:.2f} mm/s")

    st.divider()

    # --- Jauge santé ---
    if health_score is not None:
        col_gauge, col_info = st.columns([1, 2])
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Indice de Santé Global", "font": {"size": 18}},
                delta={"reference": 100},
                gauge={
                    "axis":  {"range": [0, 100], "tickwidth": 1},
                    "bar":   {"color": health_color(health_score)},
                    "steps": [
                        {"range": [0,  40], "color": "#4a0000"},
                        {"range": [40, 65], "color": "#4a2800"},
                        {"range": [65, 100],"color": "#003300"},
                    ],
                    "threshold": {
                        "line":  {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": 65,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_info:
            st.subheader(f"État : {risk_emoji(risk)} {risk}")
            if risk == "CRITIQUE":
                st.error("⛔ Intervention immédiate requise. Risque de défaillance élevé.")
            elif risk == "ALERTE":
                st.warning("⚠️ Surveillance renforcée recommandée. Planifier une inspection.")
            else:
                st.success("✅ Transformateur en bon état de fonctionnement.")

            if df_pred is not None:
                from src.inference import generate_alerts
                alerts = generate_alerts(latest)
                if alerts:
                    st.subheader("Alertes actives :")
                    for a in alerts:
                        cls = "alert-critique" if a["severity"] == "CRITIQUE" else "alert-warning"
                        st.markdown(
                            f'<div class="{cls}">🔔 {a["message"]}</div>',
                            unsafe_allow_html=True,
                        )

    st.divider()

    # --- Évolution de l'indice de santé ---
    if df_pred is not None and "health_score" in df_pred.columns:
        st.subheader("Évolution de l'indice de santé")
        fig_hi = go.Figure()
        fig_hi.add_trace(go.Scatter(
            x=df_pred.index, y=df_pred["health_score"],
            name="Indice de santé", fill="tozeroy",
            line=dict(color="#4fc3f7", width=2),
            fillcolor="rgba(79,195,247,0.1)",
        ))
        fig_hi.add_hline(y=65, line_dash="dash", line_color="#ff9100", annotation_text="Seuil ALERTE")
        fig_hi.add_hline(y=40, line_dash="dash", line_color="#ff1744", annotation_text="Seuil CRITIQUE")
        fig_hi.update_layout(
            yaxis_range=[0, 105], height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
            font_color="white", xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        )
        st.plotly_chart(fig_hi, use_container_width=True)

    # --- Charge et température ---
    st.subheader("Charge et températures")
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=df_view.index, y=df_view["load_kva"],
                                   name="Charge (kVA)", yaxis="y2",
                                   line=dict(color="#b0bec5", width=1, dash="dot")))
    fig_main.add_trace(go.Scatter(x=df_view.index, y=df_view["top_oil_temp"],
                                   name="Temp. Huile (°C)", line=dict(color="#ff9100")))
    fig_main.add_trace(go.Scatter(x=df_view.index, y=df_view["hot_spot_temp"],
                                   name="Point Chaud (°C)", line=dict(color="#ff1744")))
    fig_main.add_trace(go.Scatter(x=df_view.index, y=df_view["ambient_temp"],
                                   name="Ambiant (°C)", line=dict(color="#80deea", dash="dot")))
    fig_main.update_layout(
        yaxis=dict(title="Température (°C)", gridcolor="#333"),
        yaxis2=dict(title="Charge (kVA)", overlaying="y", side="right", gridcolor="#333"),
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
        font_color="white", legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_main, use_container_width=True)


# ===========================================================================
# PAGE 2 — Paramètres & Prévisions
# ===========================================================================
elif page == "📈 Paramètres & Prévisions":
    st.title("📈 Paramètres Mesurés vs Prédits")

    if pred_error:
        st.warning(f"⚠️ Modèles non disponibles ({pred_error}). Seules les données historiques sont affichées.")

    param_labels = {
        "top_oil_temp":  "Température Huile (°C)",
        "hot_spot_temp": "Point Chaud (°C)",
        "H2":            "Hydrogène H₂ (ppm)",
        "CO":            "Monoxyde de Carbone CO (ppm)",
        "C2H4":          "Éthylène C₂H₄ (ppm)",
        "CH4":           "Méthane CH₄ (ppm)",
        "moisture_ppm":  "Humidité dans l'Huile (ppm)",
        "vibration_mms": "Vibrations (mm/s)",
    }

    selected_param = st.selectbox("Paramètre à visualiser", list(param_labels.keys()),
                                   format_func=lambda x: param_labels[x])

    # Données réelles
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_view.index, y=df_view[selected_param],
        name="Valeur mesurée", line=dict(color="#4fc3f7", width=1.5),
    ))

    # Prédictions forecast
    if predictor and selected_param in predictor._forecast_models:
        try:
            preds = predictor.predict_forecast(
                df_feat_view, selected_param, horizon,
                feature_cols=predictor._feature_cols,
            )
            fig.add_trace(go.Scatter(
                x=df_feat_view.index, y=preds,
                name=f"Prédiction (horizon {horizon})",
                line=dict(color="#ff9100", width=2, dash="dash"),
            ))
        except Exception as exc:
            st.warning(f"Prévision indisponible : {exc}")

    # Ligne de seuil
    threshold_map = {
        "top_oil_temp": 100, "hot_spot_temp": 130, "H2": 150,
        "CO": 500, "C2H4": 50, "CH4": 120, "moisture_ppm": 30, "vibration_mms": 5,
    }
    if selected_param in threshold_map:
        fig.add_hline(
            y=threshold_map[selected_param],
            line_dash="dash", line_color="#ff1744",
            annotation_text=f"Seuil alarme = {threshold_map[selected_param]}",
        )

    fig.update_layout(
        title=param_labels[selected_param], height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
        font_color="white", xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Statistiques
    st.subheader("Statistiques sur la fenêtre sélectionnée")
    stats = df_view[selected_param].describe()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moyenne", f"{stats['mean']:.2f}")
    col2.metric("Maximum", f"{stats['max']:.2f}")
    col3.metric("Minimum", f"{stats['min']:.2f}")
    col4.metric("Écart-type", f"{stats['std']:.2f}")

    # Matrice de corrélation
    st.subheader("Corrélations entre paramètres")
    num_cols = list(param_labels.keys())
    corr = df_view[num_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Matrice de corrélation",
    )
    fig_corr.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_corr, use_container_width=True)


# ===========================================================================
# PAGE 3 — Analyse DGA
# ===========================================================================
elif page == "🧪 Analyse DGA":
    st.title("🧪 Analyse des Gaz Dissous (DGA)")

    col1, col2 = st.columns(2)

    # Évolution des gaz
    with col1:
        st.subheader("Évolution des gaz dissous")
        gas_colors = {"H2": "#f44336", "CO": "#ff9800", "C2H4": "#9c27b0", "CH4": "#2196f3"}
        fig_gas = go.Figure()
        for gas, color in gas_colors.items():
            if gas in df_view.columns:
                fig_gas.add_trace(go.Scatter(
                    x=df_view.index, y=df_view[gas],
                    name=gas, line=dict(color=color, width=1.5),
                ))
        fig_gas.update_layout(
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.3)", font_color="white",
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333", title="ppm"),
        )
        st.plotly_chart(fig_gas, use_container_width=True)

    # Triangle de Duval
    with col2:
        st.subheader("Zones de Duval (derniers relevés)")
        if "duval_zone" in df_feat_view.columns:
            zone_counts = df_feat_view["duval_zone"].value_counts().reset_index()
            zone_counts.columns = ["Zone", "Occurrences"]
            zone_colors = {"PD": "#f44336", "T1": "#ff9800", "T2": "#ff5722",
                           "T3": "#b71c1c", "D1": "#9c27b0", "D2": "#673ab7", "DT": "#3f51b5"}
            fig_pie = px.pie(
                zone_counts, values="Occurrences", names="Zone",
                color="Zone", color_discrete_map=zone_colors,
                title="Distribution des zones de Duval",
            )
            fig_pie.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Données de Duval non disponibles. Lancez le feature engineering.")

    # Scatter triangle de Duval
    st.subheader("Triangle de Duval — Positionnement des points")
    if all(c in df_feat_view.columns for c in ["duval_pct_CH4", "duval_pct_C2H4", "duval_zone"]):
        sample = df_feat_view.sample(min(500, len(df_feat_view)), random_state=42)
        fig_tri = px.scatter(
            sample, x="duval_pct_CH4", y="duval_pct_C2H4",
            color="duval_zone",
            hover_data={"H2": True, "CH4": True, "C2H4": True},
            title="Position dans le triangle CH₄ vs C₂H₄",
            labels={"duval_pct_CH4": "% CH₄", "duval_pct_C2H4": "% C₂H₄"},
        )
        fig_tri.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_tri, use_container_width=True)

    # TDCG
    if "TDCG" in df_feat_view.columns:
        st.subheader("Total des Gaz Combustibles Dissous (TDCG)")
        fig_tdcg = go.Figure()
        fig_tdcg.add_trace(go.Scatter(
            x=df_feat_view.index, y=df_feat_view["TDCG"],
            fill="tozeroy", name="TDCG",
            line=dict(color="#4caf50"),
            fillcolor="rgba(76,175,80,0.2)",
        ))
        # Niveaux IEEE C57.104
        for level, color, label in [
            (720, "#ff9800", "Niveau 1 — Surveiller"),
            (1920, "#ff5722", "Niveau 2 — Inspection"),
            (4630, "#f44336", "Niveau 3 — Action immédiate"),
        ]:
            fig_tdcg.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=label)
        fig_tdcg.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
            font_color="white", xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333", title="ppm"),
        )
        st.plotly_chart(fig_tdcg, use_container_width=True)


# ===========================================================================
# PAGE 4 — Alertes
# ===========================================================================
elif page == "🚨 Alertes":
    st.title("🚨 Tableau des Alertes de Maintenance")

    from src.inference import generate_alerts, ALERT_THRESHOLDS

    # Alertes sur le dernier relevé
    latest_row = df_raw.iloc[-1].to_dict()
    alerts = generate_alerts(latest_row)

    if alerts:
        st.subheader(f"⚠️ {len(alerts)} alerte(s) active(s) sur le dernier relevé")
        for a in alerts:
            cls = "alert-critique" if a["severity"] == "CRITIQUE" else "alert-warning"
            st.markdown(
                f'<div class="{cls}"><strong>{a["severity"]}</strong> — {a["message"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="alert-normal">✅ Aucune alerte — Tous les paramètres dans les normes.</div>',
                    unsafe_allow_html=True)

    st.divider()

    # Historique des dépassements de seuils
    st.subheader("Historique des dépassements de seuils")
    threshold_map = {
        "top_oil_temp":  100.0, "hot_spot_temp": 130.0, "H2":  150.0,
        "CO":            500.0, "C2H4":           50.0, "CH4": 120.0,
        "moisture_ppm":   30.0, "vibration_mms":   5.0,
    }

    exceedance_records = []
    for col, thresh in threshold_map.items():
        if col in df_view.columns:
            mask = df_view[col] > thresh
            if mask.any():
                for ts, val in df_view.loc[mask, col].items():
                    exceedance_records.append({
                        "Timestamp":  ts,
                        "Paramètre":  col,
                        "Valeur":     round(val, 2),
                        "Seuil":      thresh,
                        "Dépassement %": round((val - thresh) / thresh * 100, 1),
                    })

    if exceedance_records:
        df_exc = pd.DataFrame(exceedance_records).sort_values("Timestamp", ascending=False)
        st.dataframe(df_exc, use_container_width=True, height=400)

        # Graphique des fréquences de dépassement
        freq = df_exc["Paramètre"].value_counts().reset_index()
        freq.columns = ["Paramètre", "Occurrences"]
        fig_bar = px.bar(
            freq, x="Paramètre", y="Occurrences",
            color="Occurrences", color_continuous_scale="Reds",
            title="Fréquence des dépassements par paramètre",
        )
        fig_bar.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.success("Aucun dépassement de seuil sur la période sélectionnée.")

    # Résumé de l'indice de risque
    if df_pred is not None:
        st.divider()
        st.subheader("Distribution du niveau de risque")
        risk_counts = df_pred["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Niveau", "Heures"]
        color_map = {"CRITIQUE": "#ff1744", "ALERTE": "#ff9100", "NORMAL": "#00e676"}
        fig_risk = px.bar(risk_counts, x="Niveau", y="Heures",
                          color="Niveau", color_discrete_map=color_map)
        fig_risk.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_risk, use_container_width=True)


# ===========================================================================
# PAGE 5 — Upload CSV
# ===========================================================================
elif page == "📤 Prédiction CSV":
    st.title("📤 Prédiction en Direct depuis un Fichier CSV")

    st.markdown("""
    Uploadez un fichier CSV contenant des relevés de transformateur pour obtenir
    une prédiction de l'indice de santé et des alertes en temps réel.

    **Colonnes attendues :**
    `timestamp, load_kva, ambient_temp, top_oil_temp, hot_spot_temp, H2, CO, C2H4, CH4, moisture_ppm, vibration_mms`
    """)

    uploaded = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded, parse_dates=True)
            st.info(f"Fichier chargé : {uploaded.name} — {len(df_up)} lignes")

            # Préparation
            from src.data_loader import DataLoader
            from src.features import build_features
            from src.inference import generate_alerts

            loader = DataLoader.__new__(DataLoader)
            loader.filepath = Path(uploaded.name)
            loader.freq     = "1h"
            loader._df_raw  = loader._detect_and_set_index(df_up)

            try:
                loader.validate()
            except Exception as e:
                st.warning(f"Validation partielle : {e}")

            loader.clean()
            df_clean = loader.get_dataframe()
            df_up_feat = build_features(df_clean)

            if predictor:
                df_up_pred = predictor.predict_health(df_up_feat)

                # KPIs
                st.subheader("Résultats")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_health = df_up_pred["health_score"].mean()
                    st.metric("Santé Moyenne", f"{avg_health:.1f}%")
                with col2:
                    max_proba = df_up_pred["failure_proba"].max()
                    st.metric("Prob. Défaillance Max", f"{max_proba*100:.1f}%")
                with col3:
                    n_critical = (df_up_pred["risk_level"] == "CRITIQUE").sum()
                    st.metric("Heures CRITIQUE", str(n_critical))

                # Graphique
                fig_up = go.Figure()
                fig_up.add_trace(go.Scatter(
                    x=df_up_pred.index, y=df_up_pred["health_score"],
                    fill="tozeroy", name="Indice de Santé",
                    line=dict(color="#4fc3f7"),
                ))
                fig_up.add_hline(y=65, line_dash="dash", line_color="#ff9100")
                fig_up.add_hline(y=40, line_dash="dash", line_color="#ff1744")
                fig_up.update_layout(
                    yaxis_range=[0, 105], height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
                    font_color="white",
                )
                st.plotly_chart(fig_up, use_container_width=True)

                # Alertes sur dernier relevé
                st.subheader("Alertes sur le dernier relevé")
                alerts_up = generate_alerts(df_clean.iloc[-1])
                if alerts_up:
                    for a in alerts_up:
                        cls = "alert-critique" if a["severity"] == "CRITIQUE" else "alert-warning"
                        st.markdown(f'<div class="{cls}">{a["message"]}</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ Aucune alerte détectée.")

                # Tableau détaillé
                with st.expander("Voir le tableau de prédictions complet"):
                    display_cols = ["health_score", "failure_proba", "risk_level"] + \
                                   [c for c in df_clean.columns if c in df_up_pred.columns]
                    st.dataframe(df_up_pred[["health_score", "failure_proba", "risk_level"]],
                                 use_container_width=True)

                # Téléchargement
                csv_out = df_up_pred[["health_score", "failure_proba", "risk_level"]].to_csv()
                st.download_button(
                    "⬇️ Télécharger les prédictions CSV",
                    data=csv_out, file_name="predictions.csv", mime="text/csv",
                )
            else:
                st.error("Modèles non chargés. Lancez d'abord l'entraînement depuis la sidebar.")

        except Exception as exc:
            st.error(f"Erreur lors du traitement : {exc}")
            logger.exception("Erreur traitement CSV uploadé")

    else:
        # Exemple de format
        st.subheader("Format CSV attendu (exemple)")
        example = pd.DataFrame([{
            "timestamp":     "2024-01-01 12:00:00",
            "load_kva":      750, "ambient_temp": 15,
            "top_oil_temp":  68, "hot_spot_temp": 82,
            "H2":            45, "CO":           280,
            "C2H4":          8,  "CH4":           55,
            "moisture_ppm":  12, "vibration_mms": 2.1,
        }])
        st.dataframe(example, use_container_width=True)
        csv_ex = example.to_csv(index=False)
        st.download_button("⬇️ Télécharger le template CSV", data=csv_ex,
                           file_name="template_transformateur.csv", mime="text/csv")


# ===========================================================================
# PAGE 6 — Métriques Modèles
# ===========================================================================
elif page == "📉 Métriques Modèles":
    st.title("📉 Performance des Modèles ML")

    from src.inference import load_metrics
    metrics = load_metrics()

    if not metrics:
        st.warning("Aucune métrique disponible. Lancez l'entraînement depuis la sidebar.")
    else:
        # ---- Métriques de classification ----
        if "classifier_metrics" in metrics:
            st.subheader("🎯 Modèle de Classification (Défaillance)")
            cm = metrics["classifier_metrics"]

            col1, col2, col3 = st.columns(3)
            col1.metric("F1 (XGBoost CV)", f"{cm.get('xgb_f1_cv', 0):.3f}")
            col2.metric("AUC-ROC (XGBoost CV)", f"{cm.get('xgb_auc_cv', 0):.3f}")
            col3.metric("F1 (Random Forest CV)", f"{cm.get('rf_f1_cv', 0):.3f}")

            # Rapport de classification
            if "final_test_report" in cm:
                report = cm["final_test_report"]
                report_df = pd.DataFrame(report).T
                try:
                    report_df = report_df.drop(["accuracy"], errors="ignore")
                    report_df = report_df[["precision", "recall", "f1-score", "support"]]
                    report_df = report_df.dropna()
                    st.subheader("Rapport de classification (dernier fold test)")
                    st.dataframe(report_df.style.format("{:.3f}", subset=["precision", "recall", "f1-score"]),
                                 use_container_width=True)
                except Exception:
                    st.json(report)

            # Matrice de confusion
            if "confusion_matrix" in cm:
                conf = np.array(cm["confusion_matrix"])
                fig_cm = px.imshow(
                    conf, text_auto=True,
                    labels=dict(x="Prédit", y="Réel", color="Count"),
                    x=["Sain (0)", "Défaillance (1)"],
                    y=["Sain (0)", "Défaillance (1)"],
                    color_continuous_scale="Blues",
                    title="Matrice de Confusion",
                )
                fig_cm.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig_cm, use_container_width=True)

        # ---- Métriques de forecasting ----
        if "forecast_metrics" in metrics:
            st.subheader("📈 Modèles de Forecasting (par paramètre)")
            fm = metrics["forecast_metrics"]

            rows = []
            for target, horizons in fm.items():
                for hor, m in horizons.items():
                    rows.append({
                        "Paramètre": target,
                        "Horizon":   hor,
                        "RMSE CV":   round(m.get("cv_rmse_mean", 0), 3),
                        "±":         round(m.get("cv_rmse_std", 0), 3),
                        "MAE CV":    round(m.get("cv_mae_mean", 0), 3),
                        "N samples": m.get("n_samples", 0),
                    })

            if rows:
                df_fm = pd.DataFrame(rows)
                st.dataframe(df_fm, use_container_width=True)

                fig_rmse = px.bar(
                    df_fm, x="Paramètre", y="RMSE CV",
                    color="Horizon", barmode="group",
                    title="RMSE par paramètre et horizon",
                    error_y="±",
                )
                fig_rmse.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig_rmse, use_container_width=True)

    # Informations système
    st.divider()
    st.subheader("Informations système")
    col1, col2 = st.columns(2)
    with col1:
        model_files = list(MODELS_DIR.glob("*.pkl")) if MODELS_DIR.exists() else []
        st.write(f"**Modèles disponibles :** {len(model_files)}")
        for f in model_files:
            size_kb = f.stat().st_size / 1024
            st.write(f"  - {f.name} ({size_kb:.1f} Ko)")
    with col2:
        st.write(f"**Données :** {len(df_raw):,} relevés horaires")
        st.write(f"**Période :** {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
        st.write(f"**Features :** {df_feat.shape[1]} colonnes après engineering")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "PredictiveTransfo v1.0 — Maintenance prédictive de transformateurs HTA/BT | "
    "Conforme IEC 60076-7 / IEC 60599 / IEEE C57.104"
)
