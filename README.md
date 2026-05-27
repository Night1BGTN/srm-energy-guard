# ⚡ SRM Smart Energy Guard

> Système intelligent de surveillance énergétique — PFE Licence 2024-2025

## 🚀 Dashboard en ligne

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 📋 Description

Dashboard interactif de surveillance énergétique développé dans le cadre du PFE Licence.  
Analyse 100 clients (résidentiels, commerciaux, industriels) sur le dataset UCI ElectricityLoadDiagrams 2011-2014.

## 🗂️ Structure

```
srm-energy-guard/
├── app.py            # Application Streamlit principale
├── requirements.txt  # Dépendances Python
└── README.md         # Ce fichier
```

## 📊 Fonctionnalités

| Page | Contenu |
|------|---------|
| 🏠 Vue Globale | KPIs réseau, séries temporelles, heatmap horaire |
| 👤 Surveillance Client | Série temporelle + détection anomalies par client |
| ⚠️ Score de Risque | Carte de risque 100 clients, scatter, top 20 |
| 📈 Prédictions | Comparaison Prophet / GRU / CNN-LSTM / LSTM |
| 🔔 Alertes | Clients critiques + recommandations + synthèse fraudes |

## 🤖 Modèles développés

### Détection d'anomalies
- **Isolation Forest** — 4.62% anomalies (Rés 3% | Com 5% | Ind 8%)
- **LSTM Autoencoder** — 1,092 anomalies (3.12%) — seuil 0.002033
- **Consensus IF + AE** — 8 anomalies haute confiance

### Classification supervisée
- **Random Forest** — Accuracy 99.29% — F1=0.922 — AUC=0.999
- **XGBoost** — Accuracy 98.34% — F1=0.847 — AUC=0.999
- **LightGBM** — Accuracy 97.73% — F1=0.801 — AUC=0.999

### Prévision énergétique
- **LSTM (meilleur)** — MAE=4.23 kWh — RMSE=5.65 — **+68.8% vs Prophet**
- **CNN-LSTM** — MAE=5.27 kWh — RMSE=7.00
- **GRU** — MAE=5.32 kWh — RMSE=7.35
- **Prophet** (baseline) — MAE=13.56 kWh — RMSE=16.19

## 💻 Lancement en local

```bash
# 1. Cloner le repo
git clone https://github.com/VOTRE_USERNAME/srm-energy-guard
cd srm-energy-guard

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

Puis ouvrir [http://localhost:8501](http://localhost:8501) et uploader `srm_100clients.parquet`.

## ⚙️ Déploiement Streamlit Cloud

1. Pusher ce repo sur GitHub (public)
2. Aller sur [share.streamlit.io](https://share.streamlit.io)
3. **New app** → sélectionner ce repo → `app.py`
4. Deploy !

> ⚠️ Le fichier `srm_100clients.parquet` (84 Mo) n'est PAS inclus dans le repo.  
> Il doit être uploadé directement dans l'interface via le bouton dédié.

## 📌 Dataset

UCI ElectricityLoadDiagrams 2011-2014  
- 100 clients (50 résidentiels + 30 commerciaux + 20 industriels)  
- 3,471,435 lignes × 27 colonnes  
- Fichier : `srm_100clients.parquet` (84 Mo)
