# Projet COVID-19 : Analyse et Prédiction de Risque

Ce projet a été réalisé par le **Groupe 6** dans le cadre du cours de Data Analyst (Python & Machine Learning).

Il s'agit d'une application **Streamlit** interactive permettant d'analyser les données épidémiologiques du COVID-19 et de prédire le risque de mortalité des patients à l'aide d'un modèle de Machine Learning (Random Forest).


## 🚀 Installation et Lancement

Pour que l'application fonctionne correctement (notamment le modèle de prédiction), il est important d'installer les dépendances avec les bonnes versions.

### 1. Cloner ou télécharger le projet
Assurez-vous d'avoir tous les fichiers du projet, notamment :
- `app.py`
- `requirements.txt`
- `meilleur_modele_random_forest_compressed.pkl` (Le modèle)
- `data_cleaned_final.csv` (ou le fichier de données compressé)

### 2. Installer les dépendances
Ouvrez un terminal ou une invite de commande dans le dossier du projet et exécutez :

```bash
pip install -r requirements.txt
```

> **Note importante :** Cette commande installera `scikit-learn>=1.8.0`, ce qui est **indispensable** pour lire le modèle de prédiction inclus.

### 3. Lancer l'application
Toujours dans le terminal, lancez la commande :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut.

## 📊 Fonctionnalités

1.  **Tableau de bord** : Vue d'ensemble des statistiques (Mortalité, Pneumonie, etc.) avec graphiques interactifs.
2.  **Prédiction** : Formulaire pour entrer les symptômes d'un patient et obtenir une estimation de risque (Faible/Élevé) en temps réel.
3.  **Analyses** : Explorateur de données permettant de visualiser les distributions et corrélations.
4.  **Questions** : Réponses argumentées aux questions posées dans le cahier des charges.
5.  **Import CSV** : Possibilité de charger vos propres données pour analyse.

---
*Projet généré le 17 Janvier 2026 - Licence Pro Data Analysis*


