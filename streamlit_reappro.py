import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import plotly.express as px
import io

st.title("Système de calcul de reapprovisionnement")

# ============================================================================
# CONFIGURATION DES PARAMÈTRES (Toujours visible)
# ============================================================================

# Configuration des paramètres d'entreprise dans la sidebar
st.sidebar.header("🏪 Configuration Entreprise")

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
        max_value=52,
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
st.write("## ⚙️ Configuration des Paramètres de Réapprovisionnement")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("### 📊 Niveau de Service")
    service_level = st.selectbox(
        "Niveau de service souhaité",
        options=[90, 95, 97.5, 99],
        index=1,  # 95% par défaut
        help="Probabilité de ne pas avoir de rupture de stock"
    )
    
    # Correspondance niveau de service -> coefficient Z
    z_values = {90: 1.28, 95: 1.65, 97.5: 1.96, 99: 2.33}
    z = z_values[service_level]
    
    st.metric("Coefficient Z", f"{z}")

with col2:
    st.write("### 🚚 Délai de Livraison")
    lead_time = st.number_input(
        "Délai de livraison (jours)",
        min_value=1,
        max_value=90,
        value=21,
        help="Temps entre la commande et la réception"
    )
    
    st.metric("Délai configuré", f"{lead_time} jours")

with col3:
    st.write("### 📅 Horizon de Prévision")
    forcast = st.number_input(
        "Horizon de prévision (jours)",
        min_value=7,
        max_value=120,
        value=30,
        help="Période pour laquelle calculer le stock"
    )
    
    st.metric("Horizon configuré", f"{forcast} jours")

# Affichage des paramètres configurés
st.info(f"""
**Paramètres configurés :**
- Niveau de service : {service_level}% (Z = {z})
- Délai de livraison : {lead_time} jours
- Horizon de prévision : {forcast} jours
""")

st.divider()

# ============================================================================
# UPLOAD ET TRAITEMENT DES FICHIERS
# ============================================================================

# Upload three files
uploaded_file1 = st.file_uploader("Choisissez un fichier (Achats_vente) sur votre ordinateur", type=["xlsx"], key="sales")
uploaded_file2 = st.file_uploader("Choisissez un fichier (stock) sur votre ordinateur", type=["xlsx"], key="stock")
uploaded_file3 = st.file_uploader("Choisissez un fichier (cde_spécial) sur votre ordinateur", type=["xlsx"], key="service")

if uploaded_file1 and uploaded_file2 and uploaded_file3:
    try:
        # Read the files into DataFrames
        df_sales = pd.read_excel(uploaded_file1)
        df_stock = pd.read_excel(uploaded_file2)
        df_service = pd.read_excel(uploaded_file3)

        st.success("All files uploaded successfully!")

        # Merge sales and stock on 'Code Article'
        df_m = df_sales.merge(df_stock[['Code Article','Qte En Stock - FDM', 'Qte En Stock - FDM.1']], on='Code Article', how='left')
        df_sales['2023 Qté en stock'] = df_m['Qte En Stock - FDM']
        df_sales['2024 Qté en stock'] = df_m['Qte En Stock - FDM.1']
        # Merge with service file
        df_spe = df_sales.merge(df_service[['Code Article', '2023', '2024', '2025']], on= 'Code Article', how ='left').fillna(0)
        df_spe['2023 Qté VE'] = df_spe['2023 Qté Vendue'] - df_spe['2023']
        df_spe['2024 Qté VE'] = df_spe['2024 Qté Vendue'] - df_spe['2024']
        df_spe['2025 Qté VE'] = df_spe['2025 Qté Vendue'] - df_spe['2025']

        st.write("Aperçu des données fusionnées:")
        # You can now proceed with further analysis on merged_df
    except Exception as e:
        st.error(f"Un Erreur est survenuelors de la ficher : {e}")
else:
    st.info("Veuillez télécharger les trois fichiers pour continuer.")

# remplacer les valeur Nan par 0
df_spe['2023 Qté en stock'] = df_spe['2023 Qté en stock'].fillna(0)
df_spe['2024 Qté en stock'] = df_spe['2024 Qté en stock'].fillna(0)
df_spe['2025 Qté en stock'] = df_spe['2025 Qté en stock'].fillna(0)
df_spe['2023 Qté Reçue '].replace('', 0, inplace=True)
df_spe['2023 Qté Reçue '] = df_spe['2023 Qté Reçue '].fillna(0)

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
jours_commerciaux = int(jours_ouverture_bruts - jours_feries_fermeture)
jours_ecoules_2025 = (pd.to_datetime(current_day) - pd.to_datetime("2025-01-01")).days - 40

# 1. PÉRIODES COHÉRENTES - Jours d'ouverture réels de votre entreprise
st.write("### 1. PÉRIODES COHÉRENTES - Jours d'ouverture réels de votre entreprise")

st.markdown(f"""
🏪 **CALCUL SPÉCIFIQUE À VOTRE ENTREPRISE :**
- Jours/semaine : {jours_par_semaine} (incluant samedi matin)
- Jours bruts/an : {jours_ouverture_bruts}
- Jours fériés fermés : {jours_feries_fermeture}
- Congés : Pris en alternance (pas de fermeture)
- Jours commerciaux effectifs : {jours_commerciaux}
""")

st.write(f"📅 Jours écoulés en 2025 : {jours_ecoules_2025}")

# 2. CALCUL DES VENTES QUOTIDIENNES (périodes cohérentes)
df_filtre['ventes_quotidiennes_2023'] = df_filtre['2023 Qté VE'] / jours_commerciaux
df_filtre['ventes_quotidiennes_2024'] = df_filtre['2024 Qté VE'] / jours_commerciaux
df_filtre['ventes_quotidiennes_2025'] = df_filtre['2025 Qté VE'] / jours_ecoules_2025

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

# Graphique classer le vente quotidiennes par chaque marque dans l'order decoissant
brand_sales_ranking = df_filtre.groupby('Code Article')['ventes_quotidiennes_moyennes'].mean().sort_values(ascending=False).head(20)
# diagramme top 20 artcile des ventes quotidiennes moyennes
plt.figure(figsize=(10, 6))
# Access the index for the x-axis values
sns.barplot(x=brand_sales_ranking.index, y=brand_sales_ranking.values)
plt.xlabel('Code Article') # Changed label to reflect the x-axis content
plt.ylabel('Ventes quotidiennes moyennes')
plt.title('Top 20 des articles par ventes quotidiennes moyennes') # Changed title to reflect article instead of brand
plt.xticks(rotation=90)
st.pyplot(plt)

# 1. Trier les articles par ventes quotidiennes moyennes décroissantes
df_filtre = df_filtre.sort_values(by='ventes_quotidiennes_moyennes', ascending=False)

# 2. Calculer le pourcentage de chaque article
total_sales = df_filtre['ventes_quotidiennes_moyennes'].sum()
df_filtre['pourcentage'] = (df_filtre['ventes_quotidiennes_moyennes'] / total_sales) * 100

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

# 5. Afficher la répartition
st.markdown("Quantité de chaque catégorie :")
st.write(df_filtre['categorie'].value_counts())

#Pie chart catégorielle par nombre de ventes
categorie_count = df_filtre['categorie'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(categorie_count, labels=categorie_count.index, autopct='%1.1f%%', startangle=140)
plt.title('Répartition des articles par catégorie')
plt.axis('equal')
st.pyplot(plt)

# Intergrer avec tableau de prix
uploaded_filed4 = st.file_uploader("Choisissez un fichier (prix) sur votre ordinateur", type=["xlsx"], key="prix")
if uploaded_filed4:
    df_prix = pd.read_excel(uploaded_filed4)

    merged_df = pd.merge(df_filtre, df_prix, on='Code Article', how='left')
    merged_df.drop(columns=['Marque_y'], inplace=True)
    st.dataframe(merged_df)

#filtre les marques seulment stock par ottogo
brands = ['APLUS', 'ASSO', 'AUTOPART', 'AVA COOLING', 'BARDAHL', 'BERU', 'BRILLANT TOOLS', 'CALORSTAT', 'CASTROL', 'CHAMPION LUBRICANTS', 'CLAS', 'CORTECO MEILLOR',
          'DAYCO', 'DELPHI AUTOMOTIVE', 'EBI', 'ELRING', 'FACET', 'FERODO', 'GRAF', 'GTI SODIFAC', 'GUTTMANN', 'HIDRIA', 'Hitachi', 'INTFRADIS', 'IZAR', 'KODAK', 'KSTOOLS',
          'LIQUIMOLY', 'LUK', 'MAX BATTERIE', 'MEAT and Doria', 'MECAFILTER', 'MISFAT', 'MOBIL', 'MONROE', 'NEOLUX', 'NEXANS', 'NGK', 'OPTIMA', 'OSRAM', 'PIERBURG',
          'PROXITECH', 'REBORN', 'RYMEC', 'SACHS', 'SASIC', 'SICAD', 'SIIL INDUSTRIE', 'SNR', 'SNRA', 'SPILU', 'STABILUS', 'STECO', 'T.R.W', 'TC MATIC', 'TECHNIKIT',
          'TMI', 'TOTAL', 'TRISCAN', 'UPOLL', 'VALEO', 'Valvoline', 'WARM UP']
print(len(brands))
ottogo_stock = merged_df[merged_df['Marque_x'].isin(brands)]
ottogo_stock.fillna(0, inplace=True)

columns_to_check = ['2024 Qté Reçue', '2024 Qté Vendue', '2024 Qté en stock', '2025 Qté Reçue', '2025 Qté Vendue', 
                    '2023 achat total', '2023 vente total', '2024 achat total', '2024 vente total', '2025 achat total', 
                    '2025 vente total', '2025 Qté en stock']
# Check if all values in the columns_to_check are 0
df_non_null = ottogo_stock[~(ottogo_stock[columns_to_check] ==0).all(axis=1)]
df_non_null = df_non_null.copy()  # Ajoutez ceci avant les calculs

for annee in ['2023', '2024', '2025']:
    vente_total = f"{annee} vente total"
    achat_total = f"{annee} achat total"
    qte_vendue = f"{annee} Qté Vendue"
    qte_recue = f"{annee} Qté Reçue" if annee != '2023' else "2023 Qté Reçue "  # attention à l'espace
    qte_ve = f"{annee} Qté VE"

    df_non_null[f"marge/pcs {annee}"] = (df_non_null[vente_total] / df_non_null[qte_vendue].replace(0, np.nan)) - (df_non_null[achat_total] / df_non_null[qte_recue].replace(0, np.nan))
    df_non_null[f"marge/pcs {annee}"] = df_non_null[f"marge/pcs {annee}"].fillna(0)
    df_non_null[f"marge moyen {annee}"] = df_non_null[f"marge/pcs {annee}"] * df_non_null[qte_ve].fillna(0)

    df_non_null[f"marge total {annee}"] = df_non_null[vente_total] - df_non_null[achat_total] 

current_day = '2025-06-05'
days = (pd.to_datetime(current_day) - pd.to_datetime("2025-01-01")).days +1
df_non_null['marge prevu 2025'] = (df_non_null['marge moyen 2025']*(270/days)).fillna(0)

df_non_null['ratio_rotation_2025'] = np.where(df_non_null['2025 Qté en stock'] > 0, df_non_null['2025 Qté Vendue'] / df_non_null['2025 Qté en stock'], 0)
df_non_null['valeur_stock_2025'] = (df_non_null['2025 Qté en stock']) * abs(df_non_null['2025 achat total']/df_non_null['2025 Qté Reçue'])
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
st.markdown("Nombre d'article de chaque catégorie de stock :")
st.write(df_non_null['risque_rup'].value_counts())

# Identifier les marques perdus
a1=df_non_null['Marque_x'].unique().tolist()
set1 = set(df_non_null['Marque_x'])
set2 = set(brands)
dff1 = set1 - set2
dff2 = set2 - set1
different_words = dff1.union(dff2)
st.write("Les marques perdus pendant le nettoyagege, la raison est que toutes ces articles ne reprentent pas une valeur importante soit en quantité soit en VA ou en marge:", different_words, divider = True)

# After df_non_null is created or updated
output = io.BytesIO()
df_non_null.to_excel(output, index=False)
output.seek(0)

st.download_button(
    label="Télécharger le fichier Excel",
    data=output,
    file_name="df_non_null.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)