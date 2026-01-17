# =====================================================
# APPLICATION STREAMLIT COVID-19 - VERSION SIMPLIFIÉE
# =====================================================


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from io import BytesIO

# ------------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ------------------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Prédicteur de Risque",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# STYLE ET APPARENCE
# ------------------------------------------------------------
st.markdown("""
    <style>
    /* Fond doux */
    .stApp {
        background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 100%);
    }

    /* Titres */
    h1, h2, h3 {
        color: #0f766e !important;
        font-weight: 600;
    }

    /* Textes en vert */
    p, div, span, label, .stMarkdown, .stText {
        color: #065f46 !important;
    }

    /* Textes des métriques */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #047857 !important;
    }

    /* Texte dans les dataframes */
    .dataframe {
        color: #065f46 !important;
    }

    /* Boutons */
    .stButton > button {
        background: #14b8a6;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(20, 184, 166, 0.25);
    }
    .stButton > button:hover {
        background: #0d9488;
        transform: translateY(-2px);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ecfdf5;
        border-right: 1px solid #a7f3d0;
    }

    /* Texte de la sidebar */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #065f46 !important;
    }

    /* Inputs et selectbox */
    .stTextInput label, .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #047857 !important;
    }

    /* Caption (sous-titres) */
    .stCaption {
        color: #059669 !important;
    }
            
    </style>
""", unsafe_allow_html=True)

import zipfile

# ------------------------------------------------------------
# FONCTIONS DE CHARGEMENT DES DONNÉES
# ------------------------------------------------------------
@st.cache_data
def charger_donnees(chemin_fichier='covid19_cleaned.csv'):
    """Charge les données COVID-19 depuis un fichier CSV ou ZIP"""
    try:
        # Priorité au fichier compressé pour GitHub (plus léger)
        if os.path.exists('covid_archive.zip'):
            with zipfile.ZipFile('covid_archive.zip', 'r') as z:
                if chemin_fichier in z.namelist():
                    with z.open(chemin_fichier) as f:
                        return pd.read_csv(f)
        
        # Fallback : lecture directe si le fichier existe décompressé
        if os.path.exists(chemin_fichier):
            return pd.read_csv(chemin_fichier)
            
        st.error(f"Fichier introuvable : {chemin_fichier} (ni en direct, ni dans l'archive)")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

@st.cache_resource
def charger_modele_et_standardiseur():
    """Charge le modèle de prédiction et le standardiseur"""
    try:
        modele = joblib.load('meilleur_modele_random_forest_compressed.pkl')
        standardiseur = joblib.load('scaler.pkl')
        return modele, standardiseur
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None

def convertir_en_csv(dataframe):
    """Convertit un DataFrame en CSV pour téléchargement"""
    return dataframe.to_csv(index=False).encode('utf-8')

def convertir_en_excel(dataframe):
    """Convertit un DataFrame en Excel pour téléchargement"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Données')
    return output.getvalue()

# ------------------------------------------------------------
# CHARGEMENT DES DONNÉES ET DU MODÈLE
# ------------------------------------------------------------
donnees = charger_donnees()
modele, standardiseur = charger_modele_et_standardiseur()

# Arrêter si les données ou le modèle n'ont pas pu être chargés
if donnees is None or modele is None:
    st.stop()

# Liste des variables utilisées pour les prédictions
VARIABLES = [
    'USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'INTUBED', 'PNEUMONIA',
    'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
    'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
    'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU'
]

# ------------------------------------------------------------
# BARRE LATÉRALE - MENU DE NAVIGATION
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 COVID-19 Analyser")
    st.title("Prédicteur de Risque")

    page = st.radio("Navigation", [
        "🏠 Tableau de bord",
        "🔮 Prédiction",
        "📊 Analyses",
        "❓ Questions",
        "🤖 Modèles",
        "📁 Import CSV"
    ])

# ============================================================
# PAGE 1 : TABLEAU DE BORD
# ============================================================
if page == "🏠 Tableau de bord":
    st.title("🧬 COVID-19 - Tableau de bord")
    st.subheader("Vue d'ensemble des données")
    
    # Afficher les statistiques principales
    colonne1, colonne2, colonne3, colonne4, colonne5 = st.columns(5)
    
    colonne1.metric("Patients totaux", f"{len(donnees):,}")
    colonne2.metric("Âge moyen", f"{donnees['AGE'].mean():.0f} ans")
    colonne3.metric("Taux pneumonie", f"{(donnees['PNEUMONIA']==1).mean():.1%}")
    colonne4.metric("Taux mortalité", f"{donnees['HIGH_RISK'].mean():.1%}")
    colonne5.metric("Décès", f"{donnees['HIGH_RISK'].sum():,}")
    
    st.caption("Meilleur modèle : **Random Forest**")
    st.divider()

    # Onglets pour différentes visualisations
    onglet1, onglet2, onglet3 = st.tabs(["Distributions", "Corrélations", "Statistiques"])

    with onglet1:
        col1, col2 = st.columns(2)
        
        # Graphique 1 : Distribution de l'âge
        with col1:
            graphique = px.histogram(
                donnees, 
                x="AGE", 
                color="HIGH_RISK", 
                barmode="overlay",
                nbins=40, 
                title="Distribution de l'âge par risque"
            )
            st.plotly_chart(graphique, use_container_width=True)
        
        # Graphique 2 : Répartition du risque
        with col2:
            comptage_risque = donnees['HIGH_RISK'].value_counts()
            
            graphique = px.pie(
                values=comptage_risque.sort_index(),
                names=['Faible risque', 'Haut risque'],
                title="Répartition du risque",
                hole=0.5,
                color_discrete_sequence=["#90ee90", "#e6817c"]
            )
            graphique.update_traces(textinfo='percent+label', textposition='outside')
            st.plotly_chart(graphique, use_container_width=True)

    with onglet2:
        # Variables spécifiques pour la corrélation
        variables_correlation = [
            'MEDICAL_UNIT', 'USMER', 'PATIENT_TYPE', 'RENAL_CHRONIC', 
            'INMSUPR', 'OTHER_DISEASE', 'AGE', 'SEX', 'PNEUMONIA', 
            'DIABETES', 'COPD', 'ASTHMA', 'HIPERTENSION', 'CARDIOVASCULAR', 
            'OBESITY', 'TOBACCO', 'INTUBED', 'ICU', 'HIGH_RISK'
        ]
        
        # Garder seulement les variables qui existent dans les données
        variables_existantes = [v for v in variables_correlation if v in donnees.columns]
        
        # Calculer la matrice de corrélation
        correlations = donnees[variables_existantes].corr()
        
        # Créer le graphique de corrélation
        graphique = px.imshow(
            correlations.round(2), 
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Matrice de corrélations",
            aspect="auto",
            labels=dict(color="Corrélation")
        )
        graphique.update_layout(
            height=850, 
            margin=dict(l=100, r=20, t=100, b=20)
        )
        graphique.update_xaxes(side="top", tickangle=-45)
        
        st.plotly_chart(graphique, use_container_width=True)
        
        # Afficher les corrélations avec HIGH_RISK
        if 'HIGH_RISK' in variables_existantes:
            st.subheader("Corrélations avec le niveau de risque (HIGH_RISK)")
            correlations_risque = correlations['HIGH_RISK'].sort_values(ascending=False)
            
            # Créer un DataFrame pour un meilleur affichage
            df_corr_risque = pd.DataFrame({
                'Variable': correlations_risque.index,
                'Corrélation': correlations_risque.values
            })
            
            # Exclure HIGH_RISK lui-même et afficher top 10
            df_corr_risque = df_corr_risque[df_corr_risque['Variable'] != 'HIGH_RISK'].head(10)
            
            # Afficher le tableau
            st.dataframe(
                df_corr_risque.style.format({'Corrélation': '{:.4f}'}),
                use_container_width=True
            )
            
            # Créer un graphique à barres des top corrélations
            graphique_barres = px.bar(
                df_corr_risque,
                x='Corrélation',
                y='Variable',
                orientation='h',
                title='Top 10 corrélations avec HIGH_RISK',
                color='Corrélation',
                color_continuous_scale='RdBu_r',
                labels={'Corrélation': 'Coefficient de corrélation'}
            )
            graphique_barres.update_layout(height=400)
            st.plotly_chart(graphique_barres, use_container_width=True)

    with onglet3:
        # Statistiques descriptives
        st.dataframe(donnees[VARIABLES].describe().round(2), use_container_width=True)

# ============================================================
# PAGE 2 : PRÉDICTION INDIVIDUELLE
# ============================================================
elif page == "🔮 Prédiction":
    st.title("🔍 Évaluation du risque individuel")

    # Formulaire pour saisir les informations du patient
    with st.form("formulaire_patient"):

        col1, col2, col3 = st.columns(3)

        # Colonne 1 : Informations générales
        with col1:
            st.subheader("Informations générales")
            usmer = st.selectbox("USMER", [1, 2])
            unite_medicale = st.number_input("Unité médicale", 1, 13, 8)
            sexe = st.radio("Sexe", [1, 2], format_func=lambda x: "Femme" if x==1 else "Homme")
            age = st.slider("Âge", 0, 121, 45)
            enceinte = st.selectbox("Enceinte", [1, 2], index=1)

        # Colonne 2 : État clinique
        with col2:
            st.subheader("État clinique")
            type_patient = st.radio(
                "Type patient", 
                [1, 2],
                format_func=lambda x: "Ambulatoire" if x==1 else "Hospitalisé"
            )
            pneumonie = st.checkbox("Pneumonie")
            intubation = st.selectbox("Intubation", [0, 1, 2], index=0)
            soins_intensifs = st.selectbox("Soins intensifs", [0, 1, 2], index=0)
            classification = st.number_input("Classification finale", 1, 7, 5)

        # Colonne 3 : Antécédents médicaux
        with col3:
            st.subheader("Antécédents")
            diabete = st.checkbox("Diabète")
            copd = st.checkbox("COPD")
            asthme = st.checkbox("Asthme")
            immunodepression = st.checkbox("Immunodépression")
            hypertension = st.checkbox("Hypertension")
            cardiovasculaire = st.checkbox("Cardiovasculaire")
            obesite = st.checkbox("Obésité")
            insuffisance_renale = st.checkbox("Insuffisance rénale")
            tabac = st.checkbox("Tabac")
            autre_maladie = st.checkbox("Autre maladie")

        bouton_soumis = st.form_submit_button("Calculer le risque", type="primary")

    # Traitement de la prédiction
    if bouton_soumis:
        # Préparer les données d'entrée
        donnees_entree = np.array([[
            usmer, unite_medicale, sexe, age, enceinte, type_patient,
            1 if pneumonie else 0, intubation, soins_intensifs, classification,
            1 if diabete else 0, 1 if copd else 0, 1 if asthme else 0,
            1 if immunodepression else 0, 1 if hypertension else 0,
            1 if cardiovasculaire else 0, 1 if obesite else 0,
            1 if insuffisance_renale else 0, 1 if tabac else 0,
            1 if autre_maladie else 0
        ]])

        # Faire la prédiction
        prediction = modele.predict(donnees_entree)[0]
        probabilite = modele.predict_proba(donnees_entree)[0][1]

        # Afficher le résultat
        if prediction == 1:
            st.error(f"⚠️ RISQUE ÉLEVÉ détecté\n\nProbabilité : {probabilite*100:.2f}%")
        else:
            st.success(f"✅ Risque FAIBLE\n\nProbabilité : {probabilite*100:.2f}%")

# ============================================================
# PAGE 3 : ANALYSES EXPLORATOIRES
# ============================================================
elif page == "📊 Analyses":
    st.title("📈 Analyse Exploratoire")

    st.subheader("Statistiques descriptives")
    st.dataframe(donnees.describe().round(2))

    col1, col2 = st.columns(2)
    
    with col1:
        graphique = px.histogram(donnees, x='AGE', nbins=50, title="Distribution de l'âge")
        st.plotly_chart(graphique)
    
    with col2:
        graphique = px.pie(
            donnees, 
            names='SEX', 
            title="Répartition par sexe",
            labels={1: 'Femme', 2: 'Homme'}
        )
        st.plotly_chart(graphique)

# ============================================================
# PAGE 4 : RÉPONSES AUX QUESTIONS
# ============================================================
elif page == "❓ Questions":
    st.title("❓ Réponses aux Questions du Projet")
    st.markdown("Analyses détaillées pour chaque question")

    st.markdown("---")

    # Question 1 : Mortalité par sexe
    st.subheader("Question 1 : Mortalité par sexe")
    mortalite_par_sexe = donnees.groupby('SEX')['HIGH_RISK'].mean() * 100
    mortalite_par_sexe.index = ['Femme (1)', 'Homme (2)']

    graphique_q1 = px.bar(
        mortalite_par_sexe,
        title="Taux de mortalité par sexe",
        labels={'value': 'Taux de mortalité (%)'},
        color=mortalite_par_sexe.index,
        color_discrete_sequence=["#f1bce5", '#66b3ff']
    )
    graphique_q1.update_traces(texttemplate='%{y:.2f}%', textposition='auto')
    st.plotly_chart(graphique_q1, use_container_width=True)

    st.markdown("**Observation** : Les hommes ont un taux de mortalité plus élevé que les femmes.")
    st.markdown("---")

    # Question 2 : Hospitalisation
    st.subheader("Question 2 : Taux d'hospitalisation des patients COVID positifs")
    covid_positifs = donnees[donnees['CLASIFFICATION_FINAL'] <= 3]
    taux_hospitalisation = (covid_positifs['PATIENT_TYPE'] == 2).mean() * 100

    graphique_q2 = px.pie(
        values=[100 - taux_hospitalisation, taux_hospitalisation],
        names=['Ambulatoire', 'Hospitalisé'],
        title=f"Hospitalisation des COVID+ : {taux_hospitalisation:.1f}% hospitalisés",
        color_discrete_sequence=["#387569", "#c9adad"]
    )
    graphique_q2.update_traces(textinfo='percent+label')
    st.plotly_chart(graphique_q2, use_container_width=True)

    st.markdown("**Observation** : Environ 1 patient COVID+ sur 4 nécessite une hospitalisation.")
    st.markdown("---")

    # Question 3 : Femmes enceintes
    st.subheader("Question 3 : Mortalité chez les femmes enceintes")
    femmes_enceintes = donnees[(donnees['PREGNANT'] == 1) & (donnees['SEX'] == 1)]
    
    if len(femmes_enceintes) > 0:
        mortalite_enceintes = femmes_enceintes['HIGH_RISK'].mean() * 100
        mortalite_non_enceintes = donnees[(donnees['PREGNANT'] == 2) & (donnees['SEX'] == 1)]['HIGH_RISK'].mean() * 100

        graphique_q3 = px.bar(
            x=['Femmes enceintes', 'Femmes non enceintes'],
            y=[mortalite_enceintes, mortalite_non_enceintes],
            title="Mortalité : enceintes vs non enceintes",
            labels={'y': 'Taux de mortalité (%)'},
            color=['Enceintes', 'Non enceintes'],
            color_discrete_sequence=["#c698c0", "#cc7bd6"]
        )
        graphique_q3.update_traces(texttemplate='%{y:.2f}%', textposition='auto')
        st.plotly_chart(graphique_q3, use_container_width=True)

        st.markdown(f"**Observation** : Mortalité enceintes : **{mortalite_enceintes:.2f}%** vs **{mortalite_non_enceintes:.2f}%**")
    else:
        st.info("Aucune femme enceinte détectée dans ce jeu de données.")

    st.markdown("---")

    # Question 4 : Répartition COVID
    st.subheader("Question 4 : Répartition COVID +/-")
    pourcentage_positif = (donnees['CLASIFFICATION_FINAL'] <= 3).mean() * 100

    graphique_q4 = px.pie(
        values=[pourcentage_positif, 100 - pourcentage_positif],
        names=['Positif', 'Négatif/Inconclusif'],
        color_discrete_sequence=["#bd7777", "#92bedf"],
        title=f"Répartition COVID - {pourcentage_positif:.1f}% positif"
    )
    graphique_q4.update_traces(textinfo='percent+label', textposition='inside')
    st.plotly_chart(graphique_q4, use_container_width=True)

# ============================================================
# PAGE 5 : MODÈLES ET PERFORMANCES
# ============================================================
elif page == "🤖 Modèles":
    st.title("🤖 Modèles de Machine Learning")

    # Afficher les résultats des modèles
    if os.path.exists('resultats_modeles.csv'):
        resultats_modeles = pd.read_csv('resultats_modeles.csv')
        st.dataframe(
            resultats_modeles.style.format({
                col: '{:.4f}' for col in resultats_modeles.columns if col != 'Modèle'
            })
        )
    else:
        st.warning("Fichier resultats_modeles.csv non trouvé")

    # Afficher l'importance des variables
    if modele is not None:
        st.subheader("Importance des variables - Random Forest")
        importance = pd.Series(
            modele.feature_importances_, 
            index=VARIABLES
        ).sort_values(ascending=False)
        
        graphique = px.bar(importance.head(12), text_auto='.3f', 
                          title="Top 12 variables les plus importantes")
        st.plotly_chart(graphique)

# ============================================================
# PAGE 6 : IMPORT ET ANALYSE DE FICHIER CSV
# ============================================================
elif page == "📁 Import CSV":
    st.title("📁 Import & Analyse de Fichier CSV")
    st.markdown("Importez vos propres données CSV pour une analyse personnalisée")
    
    # Zone d'upload de fichier
    fichier_upload = st.file_uploader(
        "Choisissez un fichier CSV", 
        type=['csv'],
        help="Téléchargez un fichier CSV pour l'analyser"
    )
    
    if fichier_upload is not None:
        try:
            # Lire le fichier
            donnees_personnalisees = pd.read_csv(fichier_upload)
            
            st.success(f"✅ Fichier chargé : {fichier_upload.name}")
            
            # Informations générales
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lignes", f"{len(donnees_personnalisees):,}")
            col2.metric("Colonnes", len(donnees_personnalisees.columns))
            col3.metric("Valeurs manquantes", donnees_personnalisees.isnull().sum().sum())
            col4.metric("Taille", f"{donnees_personnalisees.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.divider()
            
            # Onglets pour différentes analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Aperçu", 
                "📊 Statistiques", 
                "📈 Visualisations",
                "🔍 Filtres",
                "💾 Export"
            ])
            
            # ONGLET 1 : Aperçu des données
            with tab1:
                st.subheader("Aperçu des données")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    nombre_lignes = st.slider("Nombre de lignes à afficher", 5, 100, 10)
                with col2:
                    mode_affichage = st.radio("Afficher", ["Début", "Fin", "Aléatoire"])
                
                # Afficher selon le choix
                if mode_affichage == "Début":
                    st.dataframe(donnees_personnalisees.head(nombre_lignes), use_container_width=True)
                elif mode_affichage == "Fin":
                    st.dataframe(donnees_personnalisees.tail(nombre_lignes), use_container_width=True)
                else:
                    st.dataframe(
                        donnees_personnalisees.sample(min(nombre_lignes, len(donnees_personnalisees))), 
                        use_container_width=True
                    )
                
                # Informations sur les types de données
                st.subheader("Types de données")
                types_donnees = pd.DataFrame({
                    'Colonne': donnees_personnalisees.columns,
                    'Type': donnees_personnalisees.dtypes.values,
                    'Valeurs manquantes': donnees_personnalisees.isnull().sum().values,
                    '% manquant': (donnees_personnalisees.isnull().sum() / len(donnees_personnalisees) * 100).round(2).values
                })
                st.dataframe(types_donnees, use_container_width=True)
            
            # ONGLET 2 : Statistiques
            with tab2:
                st.subheader("Statistiques descriptives")
                
                # Colonnes numériques
                colonnes_numeriques = donnees_personnalisees.select_dtypes(include=[np.number]).columns
                if len(colonnes_numeriques) > 0:
                    st.markdown("**Colonnes numériques**")
                    st.dataframe(
                        donnees_personnalisees[colonnes_numeriques].describe().round(2), 
                        use_container_width=True
                    )
                
                # Colonnes catégorielles
                colonnes_categorielles = donnees_personnalisees.select_dtypes(include=['object']).columns
                if len(colonnes_categorielles) > 0:
                    st.markdown("**Colonnes catégorielles**")
                    st.dataframe(
                        donnees_personnalisees[colonnes_categorielles].describe(), 
                        use_container_width=True
                    )
            
            # ONGLET 3 : Visualisations
            with tab3:
                st.subheader("Visualisations")
                
                colonnes_numeriques = donnees_personnalisees.select_dtypes(include=[np.number]).columns.tolist()
                colonnes_categorielles = donnees_personnalisees.select_dtypes(include=['object']).columns.tolist()
                
                # Graphiques pour colonnes numériques
                if len(colonnes_numeriques) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        colonne_selectionnee = st.selectbox("Colonne numérique", colonnes_numeriques)
                        if colonne_selectionnee:
                            graphique = px.histogram(
                                donnees_personnalisees, 
                                x=colonne_selectionnee, 
                                nbins=30, 
                                title=f"Distribution de {colonne_selectionnee}"
                            )
                            st.plotly_chart(graphique, use_container_width=True)
                    
                    with col2:
                        if len(colonnes_numeriques) > 1:
                            colonne_selectionnee2 = st.selectbox("Box plot", colonnes_numeriques, index=1)
                            if colonne_selectionnee2:
                                graphique = px.box(
                                    donnees_personnalisees, 
                                    y=colonne_selectionnee2, 
                                    title=f"Box plot de {colonne_selectionnee2}"
                                )
                                st.plotly_chart(graphique, use_container_width=True)
                
                # Graphiques pour colonnes catégorielles
                if len(colonnes_categorielles) > 0:
                    colonne_cat_selectionnee = st.selectbox("Colonne catégorielle", colonnes_categorielles)
                    if colonne_cat_selectionnee:
                        comptage_valeurs = donnees_personnalisees[colonne_cat_selectionnee].value_counts().head(10)
                        graphique = px.bar(
                            comptage_valeurs, 
                            title=f"Distribution de {colonne_cat_selectionnee}"
                        )
                        st.plotly_chart(graphique, use_container_width=True)
                
                # Matrice de corrélations
                if len(colonnes_numeriques) > 1:
                    st.subheader("Matrice de corrélations")
                    correlations = donnees_personnalisees[colonnes_numeriques].corr()
                    graphique = px.imshow(
                        correlations, 
                        text_auto='.2f', 
                        color_continuous_scale='RdBu_r',
                        title="Corrélations"
                    )
                    st.plotly_chart(graphique, use_container_width=True)
            
            # ONGLET 4 : Filtres
            with tab4:
                st.subheader("Filtrer les données")
                
                # Sélection des colonnes
                colonnes_selectionnees = st.multiselect(
                    "Colonnes à afficher",
                    donnees_personnalisees.columns.tolist(),
                    default=donnees_personnalisees.columns.tolist()[:5]
                )
                
                donnees_filtrees = donnees_personnalisees[colonnes_selectionnees].copy() if colonnes_selectionnees else donnees_personnalisees.copy()
                
                # Filtres numériques
                colonnes_num = donnees_filtrees.select_dtypes(include=[np.number]).columns
                for colonne in colonnes_num:
                    with st.expander(f"Filtrer {colonne}"):
                        valeur_min = float(donnees_filtrees[colonne].min())
                        valeur_max = float(donnees_filtrees[colonne].max())
                        plage = st.slider(
                            f"Plage pour {colonne}",
                            valeur_min, valeur_max, (valeur_min, valeur_max)
                        )
                        donnees_filtrees = donnees_filtrees[
                            (donnees_filtrees[colonne] >= plage[0]) & 
                            (donnees_filtrees[colonne] <= plage[1])
                        ]
                
                st.markdown(f"**{len(donnees_filtrees):,} lignes** après filtrage")
                st.dataframe(donnees_filtrees.head(50), use_container_width=True)
            
            # ONGLET 5 : Export
            with tab5:
                st.subheader("Exporter les données")
                
                col1, col2 = st.columns(2)
                
                # Export CSV
                with col1:
                    st.markdown("**Export CSV**")
                    csv = convertir_en_csv(donnees_personnalisees)
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name=f"export_{fichier_upload.name}",
                        mime="text/csv",
                    )
                
                # Export Excel
                with col2:
                    st.markdown("**Export Excel**")
                    excel = convertir_en_excel(donnees_personnalisees)
                    st.download_button(
                        label="📥 Télécharger Excel",
                        data=excel,
                        file_name=f"export_{fichier_upload.name.replace('.csv', '.xlsx')}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                
                st.divider()
                
                # Export des données filtrées
                if 'donnees_filtrees' in locals() and len(donnees_filtrees) < len(donnees_personnalisees):
                    st.markdown("**Export des données filtrées**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_filtre = convertir_en_csv(donnees_filtrees)
                        st.download_button(
                            label="📥 CSV filtré",
                            data=csv_filtre,
                            file_name=f"filtre_{fichier_upload.name}",
                            mime="text/csv",
                        )
                    
                    with col2:
                        excel_filtre = convertir_en_excel(donnees_filtrees)
                        st.download_button(
                            label="📥 Excel filtré",
                            data=excel_filtre,
                            file_name=f"filtre_{fichier_upload.name.replace('.csv', '.xlsx')}",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
    
    else:
        st.info("👆 Veuillez télécharger un fichier CSV pour commencer")
        st.markdown("""
        ### Fonctionnalités :
        - 📋 **Aperçu** : Visualisation des données
        - 📊 **Statistiques** : Analyse descriptive
        - 📈 **Visualisations** : Graphiques automatiques
        - 🔍 **Filtres** : Filtrage interactif
        - 💾 **Export** : Téléchargement CSV/Excel
        """)
# ============================================================
# FOOTER - PARTICIPANTS
# ============================================================

