# SRM Smart Energy Guard

> Systeme intelligent de surveillance energetique — PFE Licence 2024-2025

## Dashboard en ligne

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Description

Dashboard interactif de surveillance energetique developpe dans le cadre du PFE Licence.
Analyse 100 clients (residentiels, commerciaux, industriels) sur le dataset UCI ElectricityLoadDiagrams 2011-2014.

## Fonctionnalites

| Page | Contenu |
|------|---------|
| Vue Globale | KPIs reseau, series temporelles, heatmap horaire |
| Surveillance Client | Serie temporelle + detection anomalies par client |
| Score de Risque | Carte de risque 100 clients, scatter, top 20 |
| Predictions | Comparaison Prophet / GRU / CNN-LSTM / LSTM |
| Alertes | Clients critiques + recommandations + synthese fraudes |

## Modeles developpes

- Isolation Forest — 4.62% anomalies
- LSTM Autoencoder — 1,092 anomalies (3.12%)
- Random Forest — Accuracy 99.29% — AUC=0.999
- LSTM — MAE=4.23 kWh — RMSE=5.65 — +68.8% vs Prophet

## Lancement en local

pip install -r requirements.txt
streamlit run app.py

## Dataset

UCI ElectricityLoadDiagrams 2011-2014
100 clients — 3,471,435 lignes x 27 colonnes
