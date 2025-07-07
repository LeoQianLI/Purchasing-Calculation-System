# Système de Réapprovisionnement ERP

## 📋 Description

Ce projet est un système intelligent de calcul de réapprovisionnement développé avec Streamlit. Il analyse les données de vente, de stock et de commandes spéciales pour optimiser la gestion des stocks et fournir des recommandations d'achat précises.

## 🚀 Fonctionnalités

### 1. **Analyse des Données**
- Fusion automatique des données de vente, stock et commandes spéciales
- Nettoyage et traitement des données incohérentes
- Calcul des ventes quotidiennes moyennes pondérées
- Analyse statistique avec écart-type pondéré

### 2. **Classification ABC**
- Catégorisation automatique des articles selon la méthode ABC
- Analyse de Pareto (80/20) pour identifier les articles critiques
- Visualisation graphique de la répartition

### 3. **Calculs de Réapprovisionnement**
- **Stock de sécurité** : Calcul basé sur l'écart-type et le délai de livraison
- **Seuil de commande** : Point de commande optimal
- **Quantité conseil** : Suggestion de quantité à commander
- **Analyse des risques** : Identification des risques de rupture

### 4. **Analyses Avancées**
- Calcul des marges par article et par année
- Ratios de rotation des stocks
- Valorisation des stocks
- Classement multi-critères (CA, Quantité, Profit)

### 5. **Visualisations**
- Graphiques des top 20 articles par ventes
- Diagrammes circulaires de répartition ABC
- Tableaux interactifs avec filtres

## 🛠️ Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
# Cloner le repository
git clone [votre-repository-url]
cd reapprovisionnement

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Utilisation

### Lancement de l'application

```bash
streamlit run streamlit_reappro.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

### Fichiers requis

L'application nécessite **4 fichiers Excel** :

1. **Fichier Achats-Vente** (`Achat-vente.xlsx`)
   - Colonnes requises : Code Article, Marque, données de vente par année
   
2. **Fichier Stock** (`stock.xlsx`)
   - Colonnes requises : Code Article, Qte En Stock - FDM (par année)
   
3. **Fichier Commandes Spéciales** (`Q-SERVICE.xlsx`)
   - Colonnes requises : Code Article, quantités par année (2023, 2024, 2025)
   
4. **Fichier Prix** (`prix.xlsx`)
   - Colonnes requises : Code Article, informations de prix

### Workflow d'utilisation

1. **Upload des fichiers** : Téléchargez les 4 fichiers Excel requis
2. **Analyse automatique** : L'application traite et analyse les données
3. **Visualisation** : Consultez les graphiques et tableaux générés
4. **Export** : Téléchargez le fichier résultat `df_non_null.xlsx`

## 🏪 Configuration Entreprise

### Paramètres temporels
- **Jours d'ouverture** : 5.5 jours/semaine (Lun-Ven + Sam matin)
- **Jours fériés fermés** : 8 jours/an
- **Jours commerciaux effectifs** : 278 jours/an

### Paramètres de réapprovisionnement
- **Niveau de service** : 95% (Z = 1.65)
- **Délai de livraison** : 21 jours
- **Horizon de prévision** : 30 jours

### Marques gérées
Le système filtre et analyse 60+ marques spécifiques stockées par Ottogo, incluant :
APLUS, ASSO, AUTOPART, BARDAHL, CASTROL, DAYCO, FERODO, LIQUIMOLY, MOBIL, NGK, OSRAM, VALEO, etc.

## 📈 Méthodes de Calcul

### Ventes Quotidiennes Moyennes
```
Moyenne pondérée = (2023×20% + 2024×40% + 2025×40%)
```

### Stock de Sécurité
```
Stock_sécurité = Z × σ × √(délai_livraison)
```

### Seuil de Commande
```
Seuil = (Vente_quotidienne × Délai_livraison) + Stock_sécurité
```

### Quantité Conseil
```
Si Stock_actuel ≤ Seuil :
    Qté_conseil = (Vente_quotidienne × Prévision) + Stock_sécurité - Stock_actuel
Sinon :
    Qté_conseil = 0
```

## 📋 Structure des Données de Sortie

Le fichier `df_non_null.xlsx` contient :

- **Données de base** : Code Article, Marque, stocks et ventes par année
- **Calculs économiques** : Marges, CA, valorisation stocks
- **Indicateurs de performance** : Ratios de rotation, classements
- **Recommandations** : Stock de sécurité, seuils, quantités conseil
- **Analyse des risques** : Statut des risques de rupture

## 🎯 Cas d'Usage

### Pour les Responsables Achats
- Identification des articles à commander en priorité
- Calcul des quantités optimales à commander
- Analyse des risques de rupture de stock

### Pour la Direction
- Suivi des performances par marque et catégorie
- Analyse de la rotation des stocks
- Optimisation de la trésorerie

### Pour la Logistique
- Planification des réceptions
- Gestion de l'espace de stockage
- Suivi des niveaux de stock critique

## 🔧 Structure du Code

```
streamlit_reappro.py
├── Upload et fusion des données
├── Nettoyage et preprocessing
├── Calculs des ventes quotidiennes
├── Classification ABC
├── Intégration des prix
├── Filtrage par marques
├── Calculs de marges et rotations
├── Algorithmes de réapprovisionnement
├── Analyse des risques
└── Export des résultats
```

## 📁 Structure du Projet

```
reapprovisionnement/
├── streamlit_reappro.py      # Application principale
├── requirements.txt          # Dépendances Python
├── README.md                # Documentation
└── data/                    # Fichiers de données
    ├── Achat-vente1606.xlsx
    ├── stock23-241606.xlsx
    ├── Q-SERVICE.xlsx
    └── prix.xlsx
```

## 🐛 Dépannage

### Erreurs courantes

1. **Erreur de fichier manquant**
   - Vérifiez que tous les 4 fichiers sont uploadés
   - Vérifiez les noms des colonnes dans les fichiers Excel

2. **Erreur de calcul**
   - Vérifiez les formats de données (pas de texte dans les colonnes numériques)
   - Vérifiez les dates et périodes

3. **Performance lente**
   - Réduisez la taille des fichiers si possible
   - Vérifiez la mémoire disponible

## 🤝 Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créez une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez vos changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créez une Pull Request


---

**Version** : 1.0  
**Dernière mise à jour** : Janvier 2025  
**Auteur** : [Votre nom/équipe]