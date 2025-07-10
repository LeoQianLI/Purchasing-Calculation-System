import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Configuration de la page
st.set_page_config(
    page_title="📊 Système de Réapprovisionnement",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-container {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .info-container {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .warning-container {
        background: linear-gradient(90deg, #f39c12 0%, #e67e22 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .title-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .section-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Titre principal avec style
st.markdown("""
<div class="title-container">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">
        📦 Système de Calcul de Réapprovisionnement
    </h1>
    <p style="color: #f0f0f0; margin-top: 1rem; font-size: 1.2rem;">
        ✨ Optimisez votre gestion de stock avec intelligence ✨
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION DES PARAMÈTRES (Toujours visible)
# ============================================================================

# Configuration des paramètres d'entreprise dans la sidebar
st.sidebar.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
           padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
    <h2 style="color: white; margin: 0;">🏪 Configuration Entreprise</h2>
</div>
""", unsafe_allow_html=True)
with st.sidebar.expander("📅 Paramètres Temporels", expanded=True):
    jours_par_semaine = st.number_input(
        "Jours d'ouverture par semaine",
        min_value=1.0,
        max_value=7.0,
        value=5.5,
        step=0.5,
        help="Ex: 5.5 pour Lun-Ven + Sam matin"
    )
    semaines_par_an = st.number_input(
        "Semaines par an",
        min_value=48,
        max_value=54,
        value=52,
        help="Nombre de semaines d'activité par an"
    )

    jours_feries_fermeture = st.number_input(
        "Jours fériés fermés",
        min_value=0,
        max_value=20,
        value=8,
        help="Nombre de jours fériés où l'entreprise est fermée"
    )

# Configuration de la date actuelle dans la sidebar
with st.sidebar.expander("📅 Configuration Date", expanded=False):
    current_day = st.date_input(
        "Date actuelle",
        value=pd.to_datetime("2025-06-16"),
        help="Date de référence pour les calculs"
    ).strftime("%Y-%m-%d")

# Configuration des poids pour la moyenne pondérée dans la sidebar
with st.sidebar.expander("⚖️ Poids Moyenne Pondérée", expanded=False):
    st.write("Ajustez les poids pour le calcul des ventes moyennes :")

    poids_2023 = st.slider(
        "Poids 2023 (%)",
        min_value=0,
        max_value=100,
        value=20,
        help="Importance des données 2023 dans le calcul"
    ) / 100

    poids_2024 = st.slider(
        "Poids 2024 (%)",
        min_value=0,
        max_value=100,
        value=40,
        help="Importance des données 2024 dans le calcul"
    ) / 100

    poids_2025 = st.slider(
        "Poids 2025 (%)",
        min_value=0,
        max_value=100,
        value=40,
        help="Importance des données 2025 dans le calcul"
    ) / 100

    # Vérification que la somme fait 1
    total_poids = poids_2023 + poids_2024 + poids_2025
    if abs(total_poids - 1.0) > 0.01:
        st.warning(f"⚠️ La somme des poids ({total_poids:.2f}) devrait être proche de 1.0")
    else:
        st.success("✅ Poids équilibrés")

# Configuration des paramètres de réapprovisionnement (dans l'interface principale)
st.markdown("""
<div class="section-header">
    <h2 style="margin: 0;">⚙️ Configuration des Paramètres de Réapprovisionnement</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0; font-size: 1.1rem;">📊 Niveau de Service</h3>
    </div>
    """, unsafe_allow_html=True)
    
    service_level = st.selectbox(
        "🎯 Niveau de service souhaité",
        options=[90, 95, 97.5, 99],
        index=1,  # 95% par défaut
        help="Probabilité de ne pas avoir de rupture de stock"
    )

    # Correspondance niveau de service -> coefficient Z
    z_values = {90: 1.28, 95: 1.65, 97.5: 1.96, 99: 2.33}
    z = z_values[service_level]

    st.markdown(f"""
    <div class="metric-container">
        <h4 style="margin: 0; color: white;">📈 Coefficient Z</h4>
        <h2 style="margin: 0.5rem 0 0 0; color: white;">{z}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
               padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0; font-size: 1.1rem;">🚚 Délai de Livraison</h3>
    </div>
    """, unsafe_allow_html=True)
    
    lead_time = st.number_input(
        "⏰ Délai de livraison (jours)",
        min_value=1,
        max_value=90,
        value=21,
        help="Temps entre la commande et la réception"
    )

    st.markdown(f"""
    <div class="metric-container">
        <h4 style="margin: 0; color: white;">🕒 Délai configuré</h4>
        <h2 style="margin: 0.5rem 0 0 0; color: white;">{lead_time} jours</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); 
               padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0; font-size: 1.1rem;">📅 Horizon de Prévision</h3>
    </div>
    """, unsafe_allow_html=True)
    
    forcast = st.number_input(
        "🔮 Horizon de prévision (jours)",
        min_value=7,
        max_value=120,
        value=30,
        help="Période pour laquelle calculer le stock"
    )

    st.markdown(f"""
    <div class="metric-container">
        <h4 style="margin: 0; color: white;">📊 Horizon configuré</h4>
        <h2 style="margin: 0.5rem 0 0 0; color: white;">{forcast} jours</h2>
    </div>
    """, unsafe_allow_html=True)

# Affichage des paramètres configurés avec style amélioré
st.markdown(f"""
<div class="info-container">
    <h3 style="color: white; margin-bottom: 1rem;">📋 Paramètres Configurés</h3>
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
        <div style="text-align: center; margin: 0.5rem;">
            <h4 style="color: #ecf0f1; margin: 0;">🎯 Niveau de Service</h4>
            <p style="color: white; font-size: 1.2rem; margin: 0;">{service_level}% (Z = {z})</p>
        </div>
        <div style="text-align: center; margin: 0.5rem;">
            <h4 style="color: #ecf0f1; margin: 0;">🚚 Délai de Livraison</h4>
            <p style="color: white; font-size: 1.2rem; margin: 0;">{lead_time} jours</p>
        </div>
        <div style="text-align: center; margin: 0.5rem;">
            <h4 style="color: #ecf0f1; margin: 0;">📅 Horizon de Prévision</h4>
            <p style="color: white; font-size: 1.2rem; margin: 0;">{forcast} jours</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================================
# UPLOAD ET TRAITEMENT DES FICHIERS
# ============================================================================

st.markdown("""
<div class="section-header">
    <h2 style="margin: 0;">📁 Upload et Traitement des Fichiers</h2>
</div>
""", unsafe_allow_html=True)

# Créer trois colonnes pour les uploads
upload_col1, upload_col2, upload_col3 = st.columns(3)

with upload_col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
        <h4 style="color: white; margin: 0;">💰 Fichier Achats-Vente</h4>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file1 = st.file_uploader("📊 Données Achats/Vente", type=["xlsx"], key="sales")

with upload_col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
               padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
        <h4 style="color: white; margin: 0;">📦 Fichier Stock</h4>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file2 = st.file_uploader("📋 Données Stock", type=["xlsx"], key="stock")

with upload_col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); 
               padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
        <h4 style="color: white; margin: 0;">🎯 Commandes Spéciales</h4>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file3 = st.file_uploader("🔄 Commandes Spéciales", type=["xlsx"], key="service")

if uploaded_file1 and uploaded_file2 and uploaded_file3:
    try:
        # Affichage d'une barre de progression pour l'upload
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('📥 Lecture des fichiers en cours...')
        progress_bar.progress(10)
        
        # Read the files into DataFrames
        df_sales = pd.read_excel(uploaded_file1).loc[5:]
        progress_bar.progress(30)
        
        df_stock = pd.read_excel(uploaded_file2).loc[5:]
        progress_bar.progress(50)
        
        df_service = pd.read_excel(uploaded_file3).loc[4:]
        progress_bar.progress(70)
        
        status_text.text('✅ Tous les fichiers chargés avec succès!')
        progress_bar.progress(100)

        # Message de succès stylé
        st.markdown("""
        <div class="success-container">
            <h3 style="margin: 0; color: white;">🎉 Fichiers Chargés avec Succès!</h3>
            <p style="margin: 0.5rem 0 0 0; color: white;">Les trois fichiers ont été traités et sont prêts pour l'analyse.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Nettoyer les éléments de progression après un court délai
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        # Merge sales and stock on 'Code Article'
        df_m = df_sales.merge(df_stock[['Code Article','2023 Qte En Stock - FDM', '2024 Qte En Stock - FDM']], on='Code Article', how='left')
        df_sales['2023 Qté en stock'] = df_m['2023 Qte En Stock - FDM']
        df_sales['2024 Qté en stock'] = df_m['2024 Qte En Stock - FDM']
        
        # Merge with service file
        df_spe = df_sales.merge(df_service[['Code Article', 2023, 2024, 2025]], on= 'Code Article', how ='left').fillna(0)
        
        # Convert numeric columns to proper numeric types - Protection renforcée
        numeric_columns = ['2023 Qté Vendue', '2024 Qté Vendue', '2025 Qté Vendue',
                          '2023 Qté Reçue', '2024 Qté Reçue', '2025 Qté Reçue',
                          '2023 achat total', '2024 achat total', '2025 achat total',
                          '2023 vente total', '2024 vente total', '2025 vente total']
        
        for col in numeric_columns:
            if col in df_spe.columns:
                try:
                    # Conversion sécurisée avec nettoyage des valeurs infinies
                    df_spe[col] = pd.to_numeric(df_spe[col], errors='coerce').fillna(0)
                    # Remplacer les valeurs infinies par 0
                    df_spe[col] = df_spe[col].replace([np.inf, -np.inf], 0)
                except Exception as e:
                    st.warning(f"Problème de conversion pour la colonne {col}: {e}")
                    df_spe[col] = 0
        
        # Convert the special columns from service file - Protection renforcée
        try:
            df_spe[2023] = pd.to_numeric(df_spe[2023], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
            df_spe[2024] = pd.to_numeric(df_spe[2024], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
            df_spe[2025] = pd.to_numeric(df_spe[2025], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
        except Exception as e:
            st.warning(f"Problème de conversion des colonnes service: {e}")
            df_spe[2023] = 0
            df_spe[2024] = 0  
            df_spe[2025] = 0
        
        df_spe['2023 Qté VE'] = df_spe['2023 Qté Vendue'] - df_spe[2023]
        df_spe['2024 Qté VE'] = df_spe['2024 Qté Vendue'] - df_spe[2024]
        df_spe['2025 Qté VE'] = df_spe['2025 Qté Vendue'] - df_spe[2025]

        st.write("Aperçu des données fusionnées:")
        
        # remplacer les valeur Nan par 0 et convertir en numérique
        df_spe['2023 Qté en stock'] = pd.to_numeric(df_spe['2023 Qté en stock'], errors='coerce').fillna(0)
        df_spe['2024 Qté en stock'] = pd.to_numeric(df_spe['2024 Qté en stock'], errors='coerce').fillna(0)
        df_spe['2025 Qté en stock'] = pd.to_numeric(df_spe['2025 Qté en stock'], errors='coerce').fillna(0)
        df_spe['2023 Qté Reçue'] = pd.to_numeric(df_spe['2023 Qté Reçue'], errors='coerce').fillna(0)

        # filtre les valeurs negative dans la colonnes de stock
        df_filtre = df_spe[df_spe['2025 Qté en stock']>=0].copy()
        st.write("Nombre de lignes après le filtre:", len(df_filtre))

        # Transferé les valeur negative de la colonne vendu à la colonne stock
        df_filtre['2023 Qté en stock'] = df_filtre.apply(lambda x: x['2023 Qté en stock'] + abs(x['2023 Qté Vendue']) if x['2023 Qté Vendue'] < 0 else x['2023 Qté en stock'], axis=1)
        df_filtre['2024 Qté en stock'] = df_filtre.apply(lambda x: x['2024 Qté en stock'] + abs(x['2024 Qté Vendue']) if x['2024 Qté Vendue'] < 0 else x['2024 Qté en stock'], axis=1)
        df_filtre['2025 Qté en stock'] = df_filtre.apply(lambda x: x['2025 Qté en stock'] + abs(x['2025 Qté Vendue']) if x['2025 Qté Vendue'] < 0 else x['2025 Qté en stock'], axis=1)

        # Change negative values in '2025 Qté VE' to 0 and fill NaN values with 0
        df_filtre['2023 Qté VE'] = df_filtre['2023 Qté VE'].clip(lower=0).fillna(0)
        df_filtre['2024 Qté VE'] = df_filtre['2024 Qté VE'].clip(lower=0).fillna(0)
        df_filtre['2025 Qté VE'] = df_filtre['2025 Qté VE'].clip(lower=0).fillna(0)

        # calculer l'achat de 2024 les cases vide
        def achat_2024(row):
            if np.isnan(row['2024 Qté Reçue']) or row['2024 Qté Reçue'] == 0 or row['2024 Qté Reçue'] == '':
                return row['2024 Qté en stock'] - row['2023 Qté en stock'] + row['2024 Qté Vendue']
            else:
                return row['2024 Qté Reçue']
        df_filtre['2024 Qté Reçue'] = df_filtre.apply(achat_2024, axis=1)

        # calculer l'achat de 2025 pour les cases vide
        def achat_2025(row):
            if np.isnan(row['2025 Qté Reçue']) or row['2025 Qté Reçue'] == 0 or row['2025 Qté Reçue'] == '':
                return row['2025 Qté en stock'] - row['2024 Qté en stock'] + row['2025 Qté Vendue']
            else:
                return row['2025 Qté Reçue']
        df_filtre['2025 Qté Reçue'] = df_filtre.apply(achat_2025, axis=1)

        # supprimé les ligne 'Total' et 'Inconnu'
        df_filtre = df_filtre[~df_filtre['Marque'].astype(str).str.contains('Total', case=True, na=False)]
        df_filtre = df_filtre[~df_filtre['Marque'].astype(str).str.contains('(Inconnu)', case=False, na=False)]
        st.dataframe(df_filtre.head(20))

        # Calcul des variables configurées
        jours_ouverture_bruts = jours_par_semaine * semaines_par_an
        jours_commerciaux = max(1, int(jours_ouverture_bruts - jours_feries_fermeture))
        jours_ecoules_2025 = max(1, (pd.to_datetime(current_day) - pd.to_datetime("2025-01-01")).days - 40)

        # 1. PÉRIODES COHÉRENTES - Jours d'ouverture réels de votre entreprise
        st.markdown("""
        <div class="section-header">
            <h3 style="margin: 0;">📅 1. PÉRIODES COHÉRENTES - Jours d'ouverture réels de votre entreprise</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        🏪 **CALCUL SPÉCIFIQUE À L'ENTREPRISE :**
        - Jours/semaine : {jours_par_semaine} (incluant samedi matin)
        - Jours bruts/an : {jours_ouverture_bruts}
        - Jours fériés fermés : {jours_feries_fermeture}
        - Congés : Pris en alternance (pas de fermeture)
        - Jours commerciaux effectifs : {jours_commerciaux}
        """)

        st.write(f"📅 Jours écoulés en 2025 : {jours_ecoules_2025}")

        # 2. CALCUL DES VENTES QUOTIDIENNES (périodes cohérentes) - Protection contre division par zéro
        try:
            # S'assurer que les dénominateurs ne sont pas zéro
            jours_commerciaux_safe = max(1, jours_commerciaux)
            jours_ecoules_2025_safe = max(1, jours_ecoules_2025)
            
            df_filtre['ventes_quotidiennes_2023'] = df_filtre['2023 Qté VE'].fillna(0) / jours_commerciaux_safe
            df_filtre['ventes_quotidiennes_2024'] = df_filtre['2024 Qté VE'].fillna(0) / jours_commerciaux_safe
            df_filtre['ventes_quotidiennes_2025'] = df_filtre['2025 Qté VE'].fillna(0) / jours_ecoules_2025_safe
        except Exception as e:
            st.error(f"Erreur dans le calcul des ventes quotidiennes: {e}")
            df_filtre['ventes_quotidiennes_2023'] = 0
            df_filtre['ventes_quotidiennes_2024'] = 0
            df_filtre['ventes_quotidiennes_2025'] = 0

        # 3. MOYENNE PONDÉRÉE AJUSTÉE (moins de poids sur 2025 car données partielles)
        df_filtre['ventes_quotidiennes_moyennes'] = (
            df_filtre['ventes_quotidiennes_2023'] * poids_2023 +
            df_filtre['ventes_quotidiennes_2024'] * poids_2024 +
            df_filtre['ventes_quotidiennes_2025'] * poids_2025
        )

        # 4. ÉCART-TYPE PONDÉRÉ CORRECT
        df_filtre['ecart_type_pondere'] = np.sqrt(
            (poids_2023 * (df_filtre['ventes_quotidiennes_2023'] - df_filtre['ventes_quotidiennes_moyennes'])**2 +
             poids_2024 * (df_filtre['ventes_quotidiennes_2024'] - df_filtre['ventes_quotidiennes_moyennes'])**2 +
             poids_2025 * (df_filtre['ventes_quotidiennes_2025'] - df_filtre['ventes_quotidiennes_moyennes'])**2)
        )

        st.markdown(f"""
        ✅ **Périodes utilisées :**
        - 2023 & 2024: {jours_commerciaux} jours commerciaux
        - 2025: {jours_ecoules_2025} jours écoulés

        ✅ **Poids ajustés :** 2023({poids_2023}), 2024({poids_2024}), 2025({poids_2025})
        """)

        # Section pour les graphiques
        st.markdown("""
        <div class="section-header">
            <h3 style="margin: 0;">📊 Analyse des Ventes Quotidiennes</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Graphique interactif avec Plotly pour les ventes quotidiennes
        brand_sales_ranking = df_filtre.groupby('Code Article')['ventes_quotidiennes_moyennes'].mean().sort_values(ascending=False).head(20)
        
        # Créer un graphique interactif avec Plotly
        fig_sales = go.Figure(data=[
            go.Bar(
                x=brand_sales_ranking.index,
                y=brand_sales_ranking.values,
                marker=dict(
                    color=brand_sales_ranking.values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Ventes/jour")
                ),
                text=[f'{val:.2f}' for val in brand_sales_ranking.values],
                textposition='auto',
            )
        ])
        
        fig_sales.update_layout(
            title={
                'text': '🏆 Top 20 des Articles par Ventes Quotidiennes Moyennes',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title='📦 Code Article',
            yaxis_title='📈 Ventes Quotidiennes Moyennes',
            template='plotly_white',
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_sales, use_container_width=True)

        # 1. Trier les articles par ventes quotidiennes moyennes décroissantes
        df_filtre = df_filtre.sort_values(by='ventes_quotidiennes_moyennes', ascending=False)

        # 2. Calculer le pourcentage de chaque article - Protection complète
        try:
            total_sales = df_filtre['ventes_quotidiennes_moyennes'].fillna(0).sum()
            if total_sales == 0 or pd.isna(total_sales):
                st.markdown("""
                <div class="warning-container">
                    <h4 style="margin: 0; color: white;">⚠️ Attention</h4>
                    <p style="margin: 0.5rem 0 0 0; color: white;">Aucune vente détectée, impossible de calculer les pourcentages</p>
                </div>
                """, unsafe_allow_html=True)
                df_filtre['pourcentage'] = 0
            else:
                df_filtre['pourcentage'] = (df_filtre['ventes_quotidiennes_moyennes'].fillna(0) / total_sales) * 100
        except Exception as e:
            st.error(f"❌ Erreur dans le calcul des pourcentages: {e}")
            df_filtre['pourcentage'] = 0

        # 3. Calculer le pourcentage cumulé
        df_filtre['cum_pourcentage'] = df_filtre['pourcentage'].cumsum()

        # 4. Catégorisation ABC
        def categorielle(x):
            if x <= 20:
                return 'A'
            elif 20 < x < 80:
                return 'B'
            elif 80 <= x <= 99:
                return 'C'
            else:
                return 'D'

        df_filtre['categorie'] = df_filtre['cum_pourcentage'].apply(categorielle)

        # 5. Afficher la répartition avec style amélioré
        st.markdown("""
        <div class="section-header">
            <h3 style="margin: 0;">🎯 Analyse ABC - Répartition des Articles</h3>
        </div>
        """, unsafe_allow_html=True)
        
        categorie_count = df_filtre['categorie'].value_counts()
        
        # Afficher les métriques de répartition
        col1, col2, col3, col4 = st.columns(4)
        for i, (cat, count) in enumerate(categorie_count.items()):
            with [col1, col2, col3, col4][i]:
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="margin: 0; color: white;">📊 Catégorie {cat}</h4>
                    <h2 style="margin: 0.5rem 0 0 0; color: white;">{count}</h2>
                    <p style="margin: 0; color: #ecf0f1; font-size: 0.9rem;">articles</p>
                </div>
                """, unsafe_allow_html=True)

        # Graphique en secteurs interactif avec Plotly
        colors = ['#667eea', '#56ab2f', '#f39c12', '#e74c3c']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=categorie_count.index,
            values=categorie_count.values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent+value',
            textfont=dict(size=14),
        )])
        
        fig_pie.update_layout(
            title={
                'text': '🎯 Répartition des Articles par Catégorie ABC',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

        # Section intégration avec tableau de prix
        st.markdown("""
        <div class="section-header">
            <h3 style="margin: 0;">💰 Intégration des Données de Prix (Optionnel)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_filed4 = st.file_uploader("📊 Fichier Prix (optionnel)", type=["xlsx"], key="prix", 
                                          help="Ajoutez ce fichier pour enrichir l'analyse avec les données de prix")
        if uploaded_filed4:
            try:
                df_prix = pd.read_excel(uploaded_filed4).loc[3:]
                merged_df = pd.merge(df_filtre, df_prix, on='Code Article', how='left')
                if 'Marque_y' in merged_df.columns:
                    merged_df.drop(columns=['Marque_y'], inplace=True)
                
                st.markdown("""
                <div class="success-container">
                    <h4 style="margin: 0; color: white;">✅ Données de Prix Intégrées</h4>
                    <p style="margin: 0.5rem 0 0 0; color: white;">Les données de prix ont été fusionnées avec succès!</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(merged_df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erreur lors de l'intégration des prix: {e}")
                merged_df = df_filtre
        else:
            merged_df = df_filtre
            st.info("💡 Aucun fichier de prix fourni. Utilisation des données de base.")

        #filtre les marques seulment stock par ottogo
        brands = ['APLUS', 'ASSO', 'AUTOPART', 'AVA COOLING', 'BARDAHL', 'BERU', 'BRILLANT TOOLS', 'CALORSTAT', 'CASTROL', 'CHAMPION LUBRICANTS', 'CLAS', 'CORTECO MEILLOR',
                  'DAYCO', 'DELPHI AUTOMOTIVE', 'EBI', 'ELRING', 'FACET', 'FERODO', 'GRAF', 'GTI SODIFAC', 'GUTTMANN', 'HIDRIA', 'Hitachi', 'INTFRADIS', 'IZAR', 'KODAK', 'KSTOOLS',
                  'LIQUIMOLY', 'LUK', 'MAX BATTERIE', 'MEAT and Doria', 'MECAFILTER', 'MISFAT', 'MOBIL', 'MONROE', 'NEOLUX', 'NEXANS', 'NGK', 'OPTIMA', 'OSRAM', 'PIERBURG',
                  'PROXITECH', 'REBORN', 'RYMEC', 'SACHS', 'SASIC', 'SICAD', 'SIIL INDUSTRIE', 'SNR', 'SNRA', 'SPILU', 'STABILUS', 'STECO', 'T.R.W', 'TC MATIC', 'TECHNIKIT',
                  'TMI', 'TOTAL', 'TRISCAN', 'UPOLL', 'VALEO', 'Valvoline', 'WARM UP']
        
        marque_col = 'Marque_x' if 'Marque_x' in merged_df.columns else 'Marque'
        ottogo_stock = merged_df[merged_df[marque_col].isin(brands)]
        ottogo_stock = ottogo_stock.fillna(0)

        columns_to_check = ['2024 Qté Reçue', '2024 Qté Vendue', '2024 Qté en stock', '2025 Qté Reçue', '2025 Qté Vendue', 
                            '2023 achat total', '2023 vente total', '2024 achat total', '2024 vente total', '2025 achat total', 
                            '2025 vente total', '2025 Qté en stock']
        # Check if all values in the columns_to_check are 0
        df_non_null = ottogo_stock[~(ottogo_stock[columns_to_check] ==0).all(axis=1)]
        df_non_null = df_non_null.copy()  # Ajoutez ceci avant les calculs

        for annee in [2023, 2024, 2025]:
            vente_total = f"{annee} vente total"
            achat_total = f"{annee} achat total"
            qte_vendue = f"{annee} Qté Vendue"
            qte_recue = f"{annee} Qté Reçue" if annee != 2023 else "2023 Qté Reçue"  # attention à l'espace
            qte_ve = f"{annee} Qté VE"

            # Calcul sécurisé des marges pour éviter division par zéro
            try:
                vente_prix = np.where(df_non_null[qte_vendue] != 0, df_non_null[vente_total] / df_non_null[qte_vendue], 0)
                achat_prix = np.where(df_non_null[qte_recue] != 0, df_non_null[achat_total] / df_non_null[qte_recue], 0)
                df_non_null[f"marge/pcs {annee}"] = vente_prix - achat_prix
            except:
                df_non_null[f"marge/pcs {annee}"] = 0
            
            df_non_null[f"marge/pcs {annee}"] = df_non_null[f"marge/pcs {annee}"].fillna(0)
            df_non_null[f"marge moyen {annee}"] = df_non_null[f"marge/pcs {annee}"] * df_non_null[qte_ve].fillna(0)
            df_non_null[f"marge total {annee}"] = df_non_null[vente_total] - df_non_null[achat_total] 

        current_day = '2025-06-05'
        days = max(1, (pd.to_datetime(current_day) - pd.to_datetime("2025-01-01")).days +1)
        df_non_null['marge prevu 2025'] = (df_non_null['marge moyen 2025']*(270/days)).fillna(0)

        df_non_null['ratio_rotation_2025'] = np.where(df_non_null['2025 Qté en stock'] > 0, df_non_null['2025 Qté Vendue'] / df_non_null['2025 Qté en stock'], 0)
        # Calcul sécurisé de la valeur stock pour éviter division par zéro
        try:
            prix_unitaire = np.where(
                (df_non_null['2025 Qté Reçue'] != 0) & (df_non_null['2025 Qté Reçue'].notna()),
                df_non_null['2025 achat total'] / df_non_null['2025 Qté Reçue'],
                0
            )
            df_non_null['valeur_stock_2025'] = df_non_null['2025 Qté en stock'] * np.abs(prix_unitaire)
        except:
            df_non_null['valeur_stock_2025'] = 0
        df_non_null['valeur_stock_2025'] = df_non_null['valeur_stock_2025'].fillna(0)

        # Classement top 2000 CA, QTE, Profit
        # Classement des articles selon différents critères
        classements = {
            'CA 2024': '2024 vente total',
            'QTE 2024': 'pourcentage',
            'Profit 2024': 'marge total 2024'
        }

        for nom_rang, colonne in classements.items():
            df_non_null[nom_rang] = df_non_null[colonne].rank(ascending=False, method='first').astype(int)

        # Critères de tri pour les top 2000
        top_criteres = [
            ('ratio_rotation_2025', False),
            ('CA 2024', True),
            ('QTE 2024', True),
            ('Profit 2024', True)
        ]

        # Sélectionner les top 2000 pour chaque critère
        top_dfs = [
            df_non_null.sort_values(by=col, ascending=asc).head(2000)
            for col, asc in top_criteres
        ]

        # Fusionner et supprimer les doublons
        df_non_null = pd.concat(top_dfs).drop_duplicates(subset='Code Article', keep='first')
        st.dataframe(df_non_null[df_non_null['Code Article'] == 'APL13086AP'])

        df_non_null['stock_secu'] = z* df_non_null['ecart_type_pondere'] * np.sqrt(lead_time)
        df_non_null['seuil_cde'] = (df_non_null['ventes_quotidiennes_moyennes'] * lead_time) + df_non_null['stock_secu']

        # qté conseil
        df_non_null['cde_conseil'] = np.where(
            df_non_null['2025 Qté en stock'] <= df_non_null['seuil_cde'],
            np.maximum(0, (df_non_null['ventes_quotidiennes_moyennes'] * forcast) + df_non_null['stock_secu'] - df_non_null['2025 Qté en stock']),
            0  )

        seuil_75 = np.quantile(df_non_null['ratio_rotation_2025'], 0.75)
        median = np.median(df_non_null['ratio_rotation_2025'])

        conditions = [
            df_non_null['ratio_rotation_2025'] >= seuil_75,
            (df_non_null['ratio_rotation_2025'] >= median) & (df_non_null['ratio_rotation_2025'] < seuil_75),
            df_non_null['ratio_rotation_2025'] < median
        ]

        choices = ['élevé', 'moyen', 'faible']

        df_non_null['ratio_statut'] = np.select(conditions, choices, default='faible')

        df_non_null['risque_rup'] = np.where(
            (df_non_null['2025 Qté en stock'] == 0) & (df_non_null['ventes_quotidiennes_moyennes'] > 0),
            'Urgent',
            np.where(
                df_non_null['2025 Qté en stock'] < df_non_null['stock_secu'],
                'Besoin de réapprovisionner',
                'Inventaire adéquat'
            )
        )
        # Section résultats de l'analyse de risque
        st.markdown("""
        <div class="section-header">
            <h3 style="margin: 0;">⚠️ Analyse des Risques de Rupture</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Afficher les catégories de risque avec métriques
        risque_count = df_non_null['risque_rup'].value_counts()
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        colors_risk = {'Urgent': '#e74c3c', 'Besoin de réapprovisionner': '#f39c12', 'Inventaire adéquat': '#27ae60'}
        
        for i, (risk, count) in enumerate(risque_count.items()):
            with [risk_col1, risk_col2, risk_col3][i % 3]:
                color = colors_risk.get(risk, '#667eea')
                st.markdown(f"""
                <div style="background: {color}; padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
                    <h4 style="margin: 0; color: white;">{risk}</h4>
                    <h2 style="margin: 0.5rem 0 0 0; color: white;">{count}</h2>
                    <p style="margin: 0; color: #ecf0f1; font-size: 0.9rem;">articles</p>
                </div>
                """, unsafe_allow_html=True)

        # Section marques perdues
        st.markdown("""
        <div class="section-header">
            <h3 style="margin: 0;">🔍 Analyse des Marques</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Identifier les marques perdus
        a1=df_non_null[marque_col].unique().tolist()
        set1 = set(df_non_null[marque_col])
        set2 = set(brands)
        dff1 = set1 - set2
        dff2 = set2 - set1
        different_words = dff1.union(dff2)
        
        if different_words:
            st.markdown("""
            <div class="warning-container">
                <h4 style="margin: 0; color: white;">🎯 Marques Filtrées</h4>
                <p style="margin: 0.5rem 0 0 0; color: white;">
                    La raison du filtrage est que ces articles ne représentent pas une valeur significative 
                    en quantité, CA ou marge.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"📋 Marques concernées: {', '.join(different_words)}")
        else:
            st.success("✅ Toutes les marques ont été conservées dans l'analyse.")

        # Section téléchargement avec style amélioré
        st.markdown("""
        <div class="section-header">
            <h2 style="margin: 0;">📥 Téléchargement des Résultats</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistiques finales
        total_articles = len(df_non_null)
        urgent_articles = len(df_non_null[df_non_null['risque_rup'] == 'Urgent'])
        reappro_articles = len(df_non_null[df_non_null['risque_rup'] == 'Besoin de réapprovisionner'])
        
        # Afficher les statistiques finales
        final_col1, final_col2, final_col3, final_col4 = st.columns(4)
        
        with final_col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="margin: 0; color: white;">📊 Total Articles</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: white;">{total_articles}</h2>
                <p style="margin: 0; color: #ecf0f1; font-size: 0.9rem;">analysés</p>
            </div>
            """, unsafe_allow_html=True)
        
        with final_col2:
            st.markdown(f"""
            <div style="background: #e74c3c; padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
                <h4 style="margin: 0; color: white;">🚨 Urgent</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: white;">{urgent_articles}</h2>
                <p style="margin: 0; color: #ecf0f1; font-size: 0.9rem;">articles</p>
            </div>
            """, unsafe_allow_html=True)
        
        with final_col3:
            st.markdown(f"""
            <div style="background: #f39c12; padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
                <h4 style="margin: 0; color: white;">⚠️ Réappro</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: white;">{reappro_articles}</h2>
                <p style="margin: 0; color: #ecf0f1; font-size: 0.9rem;">articles</p>
            </div>
            """, unsafe_allow_html=True)
        
        with final_col4:
            adequate_articles = total_articles - urgent_articles - reappro_articles
            st.markdown(f"""
            <div style="background: #27ae60; padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
                <h4 style="margin: 0; color: white;">✅ Adéquat</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: white;">{adequate_articles}</h2>
                <p style="margin: 0; color: #ecf0f1; font-size: 0.9rem;">articles</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Message final stylé
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: #2c3e50; margin-bottom: 1rem;">🎉 Votre analyse est terminée!</h3>
            <p style="color: #7f8c8d; margin-bottom: 2rem;">
                Téléchargez le rapport complet pour accéder à tous les détails de l'analyse de réapprovisionnement.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Préparation du fichier Excel
        output = io.BytesIO()
        df_non_null.to_excel(output, index=False)
        output.seek(0)

        # Centrer le bouton de téléchargement
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Télécharger l'Analyse Complète (Excel)",
                data=output,
                file_name=f"analyse_reapprovisionnement_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
    except Exception as e:
        st.error(f"❌ Une erreur est survenue lors du traitement des fichiers : {e}")
else:
    # Message d'accueil stylé quand aucun fichier n'est uploadé
    st.markdown("""
    <div class="info-container">
        <h3 style="color: white; margin-bottom: 1rem;">🚀 Prêt à Commencer?</h3>
        <p style="color: #ecf0f1; margin-bottom: 1rem; line-height: 1.6;">
            Pour démarrer votre analyse de réapprovisionnement, veuillez télécharger les trois fichiers requis:
        </p>
        <ul style="color: #ecf0f1; margin-bottom: 1rem; line-height: 1.8;">
            <li>📊 <strong>Fichier Achats-Vente:</strong> Données des achats et ventes</li>
            <li>📦 <strong>Fichier Stock:</strong> Inventaire actuel</li>
            <li>🔄 <strong>Fichier Commandes Spéciales:</strong> Commandes en cours</li>
        </ul>
        <p style="color: #ecf0f1; margin: 0; font-style: italic;">
            Une fois les fichiers chargés, l'analyse commencera automatiquement.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer avec style
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%); 
           border-radius: 15px; text-align: center;">
    <h3 style="color: white; margin-bottom: 1rem;">📊 Système de Réapprovisionnement Intelligent</h3>
    <p style="color: #bdc3c7; margin-bottom: 1rem;">
        Optimisez votre gestion de stock avec des analyses avancées et des recommandations personnalisées.
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 1.5rem;">
        <div style="color: #ecf0f1;">
            <strong>🎯 Analyse ABC</strong><br>
            <span style="color: #bdc3c7;">Catégorisation intelligente</span>
        </div>
        <div style="color: #ecf0f1;">
            <strong>📈 Prévisions</strong><br>
            <span style="color: #bdc3c7;">Calculs prédictifs</span>
        </div>
        <div style="color: #ecf0f1;">
            <strong>⚡ Temps Réel</strong><br>
            <span style="color: #bdc3c7;">Analyse instantanée</span>
        </div>
        <div style="color: #ecf0f1;">
            <strong>📊 Visualisations</strong><br>
            <span style="color: #bdc3c7;">Graphiques interactifs</span>
        </div>
    </div>
    <hr style="border: 0; height: 1px; background: #7f8c8d; margin: 2rem 0;">
    <p style="color: #95a5a6; margin: 0; font-size: 0.9rem;">
        💡 Développé pour optimiser votre chaîne d'approvisionnement | Version 2.0
    </p>
</div>
""", unsafe_allow_html=True)