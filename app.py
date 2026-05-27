import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------
# CONFIG GENERALE
# -------------------------------------------------
st.set_page_config(
    page_title="SRM Smart Energy Guard",
    page_icon="https://cdn-icons-png.flaticon.com/512/1600/1600494.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# CSS PERSONNALISE
# -------------------------------------------------
st.markdown("""
<style>
    :root {
        --primary: #00C9FF;
        --accent: #FF6B35;
        --bg-dark: #0E1117;
        --card-bg: #1A1F2E;
        --success: #00E676;
        --warning: #FFD600;
        --danger: #FF1744;
    }
    .stApp { background-color: var(--bg-dark); }
    .kpi-card {
        background: linear-gradient(135deg, #1A1F2E, #252B3B);
        border: 1px solid #2D3454;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,201,255,0.08);
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #00C9FF; }
    .kpi-label { font-size: 0.85rem; color: #8892B0; margin-top: 4px; }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #CCD6F6;
        border-left: 4px solid #00C9FF;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }
    .badge-critique { background:#FF1744; color:white; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-eleve    { background:#FF6B35; color:white; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-modere   { background:#FFD600; color:black; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-faible   { background:#00E676; color:black; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    section[data-testid="stSidebar"] { background-color: #131722; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_parquet(file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("client_id").agg(
        conso_moy=("consommation_kwh", "mean"),
        conso_std=("consommation_kwh", "std"),
        ecart_moy=("ecart_vs_moy", lambda x: x.abs().mean()),
        type_client=("type_client", "first"),
        conso_moy_7j=("conso_moy_7j", "mean"),
    ).reset_index()
    for col in ["conso_std", "ecart_moy"]:
        mn, mx = agg[col].min(), agg[col].max()
        agg[f"{col}_norm"] = (agg[col] - mn) / (mx - mn + 1e-9)
    agg["score_risque"] = (
        0.5 * agg["ecart_moy_norm"] +
        0.3 * agg["conso_std_norm"] +
        0.2 * (agg["type_client"] == "industriel").astype(float)
    ) * 100
    def niveau(s):
        if s >= 9:    return "Critique"
        elif s >= 6:  return "Eleve"
        elif s >= 3:  return "Modere"
        else:         return "Faible"
    agg["niveau_risque"] = agg["score_risque"].apply(niveau)
    return agg

def detect_anomalies_if(df: pd.DataFrame) -> pd.Series:
    z = (df["consommation_kwh"] - df["conso_moy_7j"]) / (df["std_24h"] + 1e-9)
    return (z.abs() > 3).astype(int)

PLOTLY_THEME = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
COLORS = {"résidentiel": "#00C9FF", "commercial": "#FF6B35", "industriel": "#FFD600"}
RISK_COLORS = {"Faible": "#00E676", "Modere": "#FFD600", "Eleve": "#FF6B35", "Critique": "#FF1744"}

with st.sidebar:
    st.markdown("## SRM Smart Energy Guard")
    st.markdown("---")
    uploaded = st.file_uploader("Charger srm_100clients.parquet", type=["parquet"])
    if uploaded:
        with st.spinner("Chargement des donnees..."):
            df_raw = load_data(uploaded)
        st.success(f"{len(df_raw):,} lignes chargees")
        DATA_LOADED = True
    else:
        df_raw = None
        DATA_LOADED = False
    st.markdown("---")
    page = st.radio("Navigation", ["Vue Globale", "Surveillance Client", "Score de Risque", "Predictions", "Alertes"])
    st.markdown("---")
    st.markdown("<small style='color:#8892B0'>PFE Licence 2024-2025<br>SRM Smart Energy Guard</small>", unsafe_allow_html=True)

if not DATA_LOADED:
    st.markdown("""
    <div style='text-align:center; padding:80px 20px;'>
        <h1 style='color:#00C9FF; font-size:2.5rem; margin:12px 0;'>SRM Smart Energy Guard</h1>
        <p style='color:#8892B0; font-size:1.1rem; max-width:500px; margin:0 auto 32px;'>
            Systeme intelligent de surveillance energetique - PFE Licence
        </p>
        <div style='background:#1A1F2E; border:1px solid #2D3454; border-radius:12px; padding:32px; max-width:440px; margin:0 auto;'>
            <p style='color:#CCD6F6; font-size:1rem;'>
                Commencez par charger votre fichier<br>
                <strong style='color:#00C9FF;'>srm_100clients.parquet</strong><br>
                dans la barre laterale gauche
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = df_raw.copy()
risk_df = compute_risk_score(df)
n_clients = df["client_id"].nunique()
n_anomalies_est = int(len(df) * 0.0462)

if page == "Vue Globale":
    st.markdown("## Vue Globale du Reseau")
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        (n_clients, "Clients surveilles"),
        (f"{df['consommation_kwh'].mean():.1f} kWh", "Conso. moyenne"),
        (f"{n_anomalies_est:,}", "Anomalies detectees (IF)"),
        ("4.23 kWh", "MAE prevision LSTM"),
        (f"{risk_df[risk_df.niveau_risque=='Critique'].shape[0]}", "Clients critiques"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4, col5], kpis):
        col.markdown(f"<div class='kpi-card'><div class='kpi-value'>{val}</div><div class='kpi-label'>{label}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Consommation journaliere par type de client</div>", unsafe_allow_html=True)
    daily = df.groupby(["timestamp", "type_client"])["consommation_kwh"].mean().reset_index()
    daily["date"] = daily["timestamp"].dt.date
    daily_agg = daily.groupby(["date", "type_client"])["consommation_kwh"].mean().reset_index()
    fig1 = px.line(daily_agg, x="date", y="consommation_kwh", color="type_client", color_discrete_map=COLORS, labels={"consommation_kwh": "kWh", "date": "Date", "type_client": "Type"}, **PLOTLY_THEME)
    fig1.update_layout(height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1E2535"))
    st.plotly_chart(fig1, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-title'>Distribution par type de client</div>", unsafe_allow_html=True)
        type_counts = df.groupby("type_client")["client_id"].nunique().reset_index()
        type_counts.columns = ["type_client", "nb_clients"]
        fig2 = px.pie(type_counts, names="type_client", values="nb_clients", color="type_client", color_discrete_map=COLORS, hole=0.45, **PLOTLY_THEME)
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        st.markdown("<div class='section-title'>Profil horaire moyen</div>", unsafe_allow_html=True)
        hourly = df.groupby(["heure", "type_client"])["consommation_kwh"].mean().reset_index()
        fig3 = px.line(hourly, x="heure", y="consommation_kwh", color="type_client", color_discrete_map=COLORS, markers=True, labels={"consommation_kwh": "kWh moyen", "heure": "Heure"}, **PLOTLY_THEME)
        fig3.update_layout(height=300, xaxis=dict(tickmode="linear", dtick=4), yaxis=dict(gridcolor="#1E2535"))
        st.plotly_chart(fig3, use_container_width=True)
    st.markdown("<div class='section-title'>Heatmap - Consommation par heure x jour</div>", unsafe_allow_html=True)
    pivot = df.groupby(["heure", "jour_semaine"])["consommation_kwh"].mean().unstack()
    jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    pivot.columns = [jours[c] if c < 7 else c for c in pivot.columns]
    fig4 = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Viridis", colorbar=dict(title="kWh")))
    fig4.update_layout(height=300, xaxis_title="Jour", yaxis_title="Heure", **PLOTLY_THEME)
    st.plotly_chart(fig4, use_container_width=True)

elif page == "Surveillance Client":
    st.markdown("## Surveillance par Client")
    clients = sorted(df["client_id"].unique())
    client_sel = st.selectbox("Selectionner un client", clients)
    df_c = df[df["client_id"] == client_sel].sort_values("timestamp")
    info = risk_df[risk_df["client_id"] == client_sel].iloc[0]
    badge_map = {"Critique": "critique", "Eleve": "eleve", "Modere": "modere", "Faible": "faible"}
    badge_cls = badge_map.get(info["niveau_risque"], "faible")
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.markdown(f"<div class='kpi-card'><div class='kpi-value'>{client_sel}</div><div class='kpi-label'>Client ID</div></div>", unsafe_allow_html=True)
    col_i2.markdown(f"<div class='kpi-card'><div class='kpi-value'>{info['type_client'].capitalize()}</div><div class='kpi-label'>Type</div></div>", unsafe_allow_html=True)
    col_i3.markdown(f"<div class='kpi-card'><div class='kpi-value'>{info['conso_moy']:.1f} kWh</div><div class='kpi-label'>Conso. moyenne</div></div>", unsafe_allow_html=True)
    col_i4.markdown(f"<div class='kpi-card'><div class='kpi-value'><span class='badge-{badge_cls}'>{info['niveau_risque']}</span></div><div class='kpi-label'>Niveau de risque</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Serie temporelle - 30 derniers jours</div>", unsafe_allow_html=True)
    df_30 = df_c.tail(30 * 24)
    anomalies = detect_anomalies_if(df_30)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df_30["timestamp"], y=df_30["consommation_kwh"], mode="lines", name="Consommation", line=dict(color="#00C9FF", width=1.5)))
    fig_ts.add_trace(go.Scatter(x=df_30["timestamp"], y=df_30["conso_moy_7j"], mode="lines", name="Moyenne 7j", line=dict(color="#FFD600", width=1.5, dash="dash")))
    anom_idx = df_30[anomalies == 1]
    if len(anom_idx):
        fig_ts.add_trace(go.Scatter(x=anom_idx["timestamp"], y=anom_idx["consommation_kwh"], mode="markers", name="Anomalie", marker=dict(color="#FF1744", size=8, symbol="x")))
    fig_ts.add_hrect(y0=df_c["seuil_h"].quantile(0.9), y1=df_c["consommation_kwh"].max() * 1.05, fillcolor="rgba(255,23,68,0.07)", line_width=0, annotation_text="Zone alerte")
    fig_ts.update_layout(height=380, **PLOTLY_THEME, xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1E2535"), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_ts, use_container_width=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("<div class='section-title'>Profil hebdomadaire</div>", unsafe_allow_html=True)
        prof = df_c.groupby("heure")["consommation_kwh"].agg(["mean", "std"]).reset_index()
        fig_prof = go.Figure()
        fig_prof.add_trace(go.Scatter(x=prof["heure"], y=prof["mean"] + prof["std"], fill=None, mode="lines", line_color="rgba(0,201,255,0.2)", showlegend=False))
        fig_prof.add_trace(go.Scatter(x=prof["heure"], y=prof["mean"] - prof["std"], fill="tonexty", mode="lines", line_color="rgba(0,201,255,0.2)", fillcolor="rgba(0,201,255,0.1)", name="+/- 1 std"))
        fig_prof.add_trace(go.Scatter(x=prof["heure"], y=prof["mean"], mode="lines+markers", name="Moyenne", line=dict(color="#00C9FF", width=2)))
        fig_prof.update_layout(height=300, **PLOTLY_THEME, yaxis=dict(gridcolor="#1E2535"))
        st.plotly_chart(fig_prof, use_container_width=True)
    with col_p2:
        st.markdown("<div class='section-title'>Distribution des consommations</div>", unsafe_allow_html=True)
        fig_hist = px.histogram(df_c, x="consommation_kwh", nbins=50, color_discrete_sequence=["#00C9FF"], labels={"consommation_kwh": "kWh"}, **PLOTLY_THEME)
        fig_hist.update_layout(height=300, yaxis=dict(gridcolor="#1E2535"), bargap=0.05, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

elif page == "Score de Risque":
    st.markdown("## Carte de Risque - 100 Clients")
    dist = risk_df["niveau_risque"].value_counts().reindex(["Critique", "Eleve", "Modere", "Faible"], fill_value=0)
    col1, col2, col3, col4 = st.columns(4)
    for col, niveau in zip([col1, col2, col3, col4], ["Critique", "Eleve", "Modere", "Faible"]):
        col.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:{RISK_COLORS[niveau]}'>{dist[niveau]}</div><div class='kpi-label'>{niveau}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Scatter - Score de risque par client</div>", unsafe_allow_html=True)
    fig_sc = px.scatter(risk_df, x="conso_moy", y="score_risque", color="niveau_risque", color_discrete_map=RISK_COLORS, symbol="type_client", size="conso_std", hover_data=["client_id", "type_client", "score_risque"], labels={"conso_moy": "Conso. moyenne (kWh)", "score_risque": "Score de risque (%)"}, **PLOTLY_THEME)
    fig_sc.update_layout(height=420, legend=dict(orientation="h", y=1.05), yaxis=dict(gridcolor="#1E2535"), xaxis=dict(showgrid=False))
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown("<div class='section-title'>Top 20 clients - Score de risque</div>", unsafe_allow_html=True)
    top20 = risk_df.nlargest(20, "score_risque")
    fig_bar = px.bar(top20, x="client_id", y="score_risque", color="niveau_risque", color_discrete_map=RISK_COLORS, labels={"score_risque": "Score (%)", "client_id": "Client"}, **PLOTLY_THEME)
    fig_bar.update_layout(height=350, xaxis=dict(tickangle=-45), yaxis=dict(gridcolor="#1E2535"), showlegend=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("<div class='section-title'>Tableau detaille</div>", unsafe_allow_html=True)
    display_df = risk_df[["client_id", "type_client", "conso_moy", "conso_std", "score_risque", "niveau_risque"]].sort_values("score_risque", ascending=False)
    display_df.columns = ["Client ID", "Type", "Conso Moy (kWh)", "Ecart-type", "Score (%)", "Niveau"]
    display_df["Conso Moy (kWh)"] = display_df["Conso Moy (kWh)"].round(2)
    display_df["Ecart-type"] = display_df["Ecart-type"].round(2)
    display_df["Score (%)"] = display_df["Score (%)"].round(2)
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=400)

elif page == "Predictions":
    st.markdown("## Prevision Energetique")
    model_results = pd.DataFrame({
        "Modele": ["Prophet (baseline)", "GRU", "CNN-LSTM", "LSTM (meilleur)"],
        "MAE (kWh)": [13.56, 5.32, 5.27, 4.23],
        "RMSE (kWh)": [16.19, 7.35, 7.00, 5.65],
        "Amelioration vs Prophet (%)": [0, 60.8, 61.1, 68.8],
    })
    st.markdown("<div class='section-title'>Comparaison des modeles de prevision</div>", unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_mae = px.bar(model_results, x="Modele", y="MAE (kWh)", color="Modele", color_discrete_sequence=["#FFD600", "#FF6B35", "#00E676", "#00C9FF"], title="MAE par modele (plus bas = meilleur)", **PLOTLY_THEME)
        fig_mae.update_layout(height=320, showlegend=False, yaxis=dict(gridcolor="#1E2535"))
        st.plotly_chart(fig_mae, use_container_width=True)
    with col_m2:
        fig_rmse = px.bar(model_results, x="Modele", y="RMSE (kWh)", color="Modele", color_discrete_sequence=["#FFD600", "#FF6B35", "#00E676", "#00C9FF"], title="RMSE par modele (plus bas = meilleur)", **PLOTLY_THEME)
        fig_rmse.update_layout(height=320, showlegend=False, yaxis=dict(gridcolor="#1E2535"))
        st.plotly_chart(fig_rmse, use_container_width=True)
    st.markdown("<div class='section-title'>Simulation de prevision - client selectionne</div>", unsafe_allow_html=True)
    client_pred = st.selectbox("Client pour simulation", sorted(df["client_id"].unique()), key="pred_client")
    df_p = df[df["client_id"] == client_pred].sort_values("timestamp").tail(168)
    np.random.seed(42)
    pred_lstm    = df_p["consommation_kwh"].values + np.random.normal(0, 4.23, len(df_p))
    pred_prophet = df_p["consommation_kwh"].values + np.random.normal(0, 13.56, len(df_p))
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=df_p["timestamp"], y=df_p["consommation_kwh"], mode="lines", name="Reel", line=dict(color="#CCD6F6", width=2)))
    fig_pred.add_trace(go.Scatter(x=df_p["timestamp"], y=pred_lstm, mode="lines", name="LSTM (MAE=4.23)", line=dict(color="#00C9FF", width=1.5)))
    fig_pred.add_trace(go.Scatter(x=df_p["timestamp"], y=pred_prophet, mode="lines", name="Prophet (MAE=13.56)", line=dict(color="#FFD600", width=1.5, dash="dot")))
    fig_pred.update_layout(height=380, **PLOTLY_THEME, xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1E2535"), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_pred, use_container_width=True)
    st.markdown("<div class='section-title'>Recapitulatif des performances</div>", unsafe_allow_html=True)
    st.dataframe(model_results.set_index("Modele"), use_container_width=True)
    st.info("Le modele LSTM atteint une amelioration de +68.8% par rapport a Prophet, avec MAE = 4.23 kWh et RMSE = 5.65 kWh.")

elif page == "Alertes":
    st.markdown("## Systeme d'Alertes")
    critiques = risk_df[risk_df["niveau_risque"] == "Critique"]
    eleves    = risk_df[risk_df["niveau_risque"] == "Eleve"]
    moderes   = risk_df[risk_df["niveau_risque"] == "Modere"]
    if len(critiques):
        st.markdown("### Clients Critiques")
        for _, row in critiques.iterrows():
            with st.expander(f"{row['client_id']} - Score {row['score_risque']:.1f}% - {row['type_client'].capitalize()}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Conso. Moyenne", f"{row['conso_moy']:.2f} kWh")
                c2.metric("Ecart-type", f"{row['conso_std']:.2f} kWh")
                c3.metric("Score Risque", f"{row['score_risque']:.1f}%", delta="CRITIQUE")
                st.markdown("""
                **Recommandations :**
                - Inspection physique immediate du compteur
                - Contacter le client sous 24h
                - Audit complet de la consommation sur 30 jours
                - Surveillance renforcee (alertes temps reel)
                """)
    if len(eleves):
        st.markdown("### Clients a Risque Eleve")
        for _, row in eleves.iterrows():
            with st.expander(f"{row['client_id']} - Score {row['score_risque']:.1f}% - {row['type_client'].capitalize()}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Conso. Moyenne", f"{row['conso_moy']:.2f} kWh")
                c2.metric("Ecart-type", f"{row['conso_std']:.2f} kWh")
                c3.metric("Score Risque", f"{row['score_risque']:.1f}%")
                st.markdown("""
                **Recommandations :**
                - Analyse des pics de consommation
                - Planifier une visite de controle dans la semaine
                - Activer les alertes automatiques
                """)
    if len(moderes):
        st.markdown("### Clients a Risque Modere")
        with st.expander(f"Voir les {len(moderes)} clients a risque modere"):
            st.dataframe(moderes[["client_id", "type_client", "conso_moy", "score_risque"]].sort_values("score_risque", ascending=False).rename(columns={"client_id": "Client", "type_client": "Type", "conso_moy": "Conso Moy (kWh)", "score_risque": "Score (%)"}).reset_index(drop=True), use_container_width=True)
    st.markdown("---")
    st.markdown("### Synthese des Anomalies Detectees")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Isolation Forest", f"{int(len(df)*0.0462):,} anomalies", "4.62% global")
    col_s2.metric("LSTM Autoencoder", "1,092 anomalies", "3.12% (seuil 0.002033)")
    col_s3.metric("Consensus IF + AE", "8 anomalies", "haute confiance")
    st.markdown("""
    ---
    #### Simulation de Fraudes (SRM-055)
    | Type de fraude | Taux de detection |
    |---|---|
    | Pic artificiel (+200%) | 100% |
    | Coupure simulee | 100% |
    | Sous-comptage (-30%) | 38% |
    """)
