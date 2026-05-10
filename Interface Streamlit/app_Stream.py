# app_streamlit_simple.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import base64
import os
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
#coding: utf-8 

# Configuration de la page
st.set_page_config(
    page_title="PRÉDICTEUR ACADÉMIQUE",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===========================================
# CHARGEMENT DES RESSOURCES
# ===========================================
@st.cache_resource
def load_model():
    """Charger le modèle ML"""
    try:
        model_data = joblib.load('model.joblib')
        return model_data['model'], True
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle: {e}")
        return None, False

def load_image(image_path, size=(400, 300)):
    """Charger et redimensionner une image"""
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = img.resize(size, Image.Resampling.LANCZOS)
            return img
        else:
            return None
    except:
        return None

# ===========================================
# CHARGEMENT DU MODÈLE ET DES IMAGES
# ===========================================
# Charger le modèle
model, model_loaded = load_model()

# Charger les images
background_img = load_image("assets/background.png", (500, 300))
valide_img = load_image("assets/valide.jpg", (350, 250))
non_valide_img = load_image("assets/non_valide.jpg", (350, 250))

# ===========================================
# FICHIER DE STYLE CSS
# ===========================================
def load_custom_css():
    """Charger le CSS personnalisé"""
    st.markdown("""
    <style>
    /* Style général */
    .main {
        padding: 0px !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Container principal */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Header stylisé */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        border: 3px solid #3B82F6;
    }
    
    .title-main {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px !important;
    }
    
    .subtitle-main {
        font-size: 1.4rem !important;
        color: #4B5563;
        margin-bottom: 20px !important;
    }
    
    /* Onglets stylisés */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: white;
        border-radius: 15px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border: 3px solid #E5E7EB;
        padding: 15px 25px;
        color: #374151;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #3B82F6, #1D4ED8) !important;
        color: white !important;
        border-color: #3B82F6 !important;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Cartes stylisées */
    .input-card {
        background: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        border: 2px solid #E5E7EB;
        margin-bottom: 20px;
    }
    
    .result-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        padding: 30px;
        border-radius: 25px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
        border: 4px solid #3B82F6;
        text-align: center;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Boutons stylisés - COULEURS VIVES */
    .stButton button {
        width: 100%;
        border-radius: 15px !important;
        height: 60px;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        transition: all 0.3s ease !important;
        border: none !important;
        background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
        color: white !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton button:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 15px 30px rgba(255, 107, 107, 0.4) !important;
        background: linear-gradient(45deg, #FF8E53, #FF6B6B) !important;
    }
    
    .secondary-button {
        background: linear-gradient(45deg, #48BB78, #38A169) !important;
    }
    
    .secondary-button:hover {
        background: linear-gradient(45deg, #38A169, #48BB78) !important;
        box-shadow: 0 15px 30px rgba(72, 187, 120, 0.4) !important;
    }
    
    .reset-button {
        background: linear-gradient(45deg, #A0AEC0, #718096) !important;
    }
    
    .reset-button:hover {
        background: linear-gradient(45deg, #718096, #A0AEC0) !important;
        box-shadow: 0 15px 30px rgba(113, 128, 150, 0.4) !important;
    }
    
    /* Inputs stylisés */
    .stNumberInput input, .stTextInput input {
        border-radius: 12px !important;
        border: 3px solid #E5E7EB !important;
        padding: 15px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        background: #F8FAFC !important;
    }
    
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2) !important;
        background: white !important;
    }
    
    /* Labels */
    .input-label {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #374151;
        margin-bottom: 8px;
        display: block;
    }
    
    /* Badges résultats */
    .success-badge {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: #10B981;
        text-shadow: 2px 2px 4px rgba(16, 185, 129, 0.2);
        margin: 20px 0;
    }
    
    .error-badge {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: #EF4444;
        text-shadow: 2px 2px 4px rgba(239, 68, 68, 0.2);
        margin: 20px 0;
    }
    
    /* Image container */
    .image-container {
        padding: 20px;
        background: yellow;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Statistiques */
    .stats-box {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        color: white;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .title-main {
            font-size: 2rem !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            font-size: 1rem !important;
            padding: 10px 15px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================================
# COMPOSANTS DE L'INTERFACE
# ===========================================
def render_header():
    """Afficher l'en-tête principal"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="main-header">
            <h1 class="title-main">🎓 PRÉDICTEUR ACADÉMIQUE</h1>
            <p class="subtitle-main">Système de Prédiction de Validation Étudiante</p>
            <p style="color: #6B7280; font-size: 1rem;">Développé par BARGO Alfred</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Afficher l'image de fond si disponible
        with col2:
            if  background_img:
                st.image(background_img, width=515)

def create_number_input(label, key, default_value=0):
    """Créer un champ de saisie numérique"""
    st.markdown(f'<div class="input-label">{label}</div>', unsafe_allow_html=True)
    return st.number_input(
        label,
        min_value=None,
        max_value=None,
        value=default_value,
        step=0.1,
        format="%.2f",
        key=f"input_{key}",
        label_visibility="collapsed"
    )

def render_prediction_result(prediction, probability, values):
    """Afficher le résultat de la prédiction"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    if prediction == 1:
        # Étudiant validé
        st.markdown('<div class="success-badge">✅ VALIDÉ</div>', unsafe_allow_html=True)
        
        if valide_img:
            st.image(valide_img, width=400)

        st.markdown("### 🎉 Félicitations ! L'étudiant a validé sa session.")
        st.markdown(f"### 🎯 Probabilité : {probability:.1f}%")
        
        # Animation de confettis
        st.balloons()
        
    else:
        # Étudiant non validé
        st.markdown('<div class="error-badge">❌ NON VALIDÉ</div>', unsafe_allow_html=True)
        
        if non_valide_img:
            st.image(non_valide_img, width=400)
        
        st.markdown("### 📚 L'étudiant n'a pas validé sa session.")
        st.markdown(f"### 🎯 Probabilité : {probability:.1f}%")
    
    # Afficher les valeurs saisies
    with st.expander("📋 Voir les valeurs saisies"):
        for matiere, valeur in values.items():
            st.write(f"**{matiere}** : {valeur:.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_manual_prediction():
    """Interface de prédiction manuelle"""
    st.markdown("## 📝 PRÉDICTION MANUELLE")
    st.markdown("Saisissez les moyennes pondérées de l'étudiant :")
    
    # Créer deux colonnes pour les inputs
    col1, col2 = st.columns(2)
    
    # Variables d'entrée (noms courts pour l'affichage)
    variables_affichage = [
        'PHYSIQUE 2',
        'MATHÉMATIQUES 2', 
        'CHIMIE 2',
        'INFORMATIQUE 2',
        'TRANSVERSALES 2',
        'SCIENCES GRAPHIQUES'
    ]
    
    variables_completes = [
        'Moyenne Pondéré PHYSIQUE 2',
        'Moyenne Pondéré MATHÉMATIQUES 2',
        'Moyenne Pondéré CHIMIE 2',
        'Moyenne Pondéré INFORMATIQUE 2',
        'Moyenne Pondéré TRANSVERSALES 2',
        'Moyenne Pondéré SCIENCES GRAPHIQUES'
    ]
    
    # Dictionnaire pour stocker les valeurs
    valeurs = {}
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        for i in range(3):
            valeurs[variables_completes[i]] = create_number_input(
                variables_affichage[i],
                variables_affichage[i],
                0.0
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        for i in range(3, 6):
            valeurs[variables_completes[i]] = create_number_input(
                variables_affichage[i],
                variables_affichage[i],
                0.0
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Calculer et afficher le total
    total = sum(valeurs.values())
    col_total1, col_total2, col_total3 = st.columns([1, 2, 1])
    
    with col_total2:
        st.markdown(f"""
        <div class="stats-box">
            <h3>📊 TOTAL DES PONDÉRATIONS</h3>
            <h1 style="font-size: 3rem; margin: 10px 0;">{total:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Boutons d'action
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        st.markdown("""
        <style>
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(45deg, #48BB78, #38A169) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        predict_button = st.button("🚀 **LANCER LA PRÉDICTION**", key="predict_manual", type="primary")
    
    with col_btn2:
        # Utiliser du CSS pour changer la couleur du bouton
        st.markdown("""
        <style>
        div[data-testid="stButton"] button[kind="secondary"] {
            background: linear-gradient(45deg, #F87171, #EF4444) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        reset_button = st.button("🔄 **RÉINITIALISER**", key="reset_manual", type="secondary")
    
    with col_btn3:
        # Utiliser du CSS pour le bouton reset
        st.markdown("""
        <style>

        div[data-testid="stButton"] button[kind="tertiary"]:nth-of-type(2) {

            background: linear-gradient(45deg, #4299E1, #3182CE) !important;

        }

        </style>
        """, unsafe_allow_html=True)
        export_button = st.button("📥 **EXPORTER**", key="export_manual", type="tertiary")
    
    # Gestion des boutons
    if reset_button:
        st.session_state.clear()
        st.rerun()
    
    if predict_button:
        if not model_loaded:
            st.error("⚠️ Modèle non chargé. Vérifiez le fichier 'model.joblib'.")
        else:
            # Validation simple
            valid = True
            for key, value in valeurs.items():
                if value is None:
                    valid = False
                    break
            
            if valid:
                with st.spinner("🔮 Analyse en cours..."):
                    try:
                        # Préparer les données
                        input_df = pd.DataFrame([valeurs])
                        
                        # Faire la prédiction
                        prediction = model.predict(input_df)[0]
                        proba = model.predict_proba(input_df)[0]
                        probability = proba[1] * 100
                        
                        # Afficher le résultat
                        render_prediction_result(prediction, probability, valeurs)
                        
                        # Stocker pour export
                        st.session_state.last_prediction = {
                            'prediction': prediction,
                            'probability': probability,
                            'values': valeurs.copy()
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            else:
                st.warning("⚠️ Veuillez remplir tous les champs.")
    
    if export_button and 'last_prediction' in st.session_state:
        # Créer un DataFrame pour l'export
        export_df = pd.DataFrame([st.session_state.last_prediction['values']])
        export_df['Prédiction'] = 'Validé' if st.session_state.last_prediction['prediction'] == 1 else 'Non validé'
        export_df['Probabilité'] = st.session_state.last_prediction['probability']
        
        # Convertir en CSV
        csv = export_df.to_csv(index=False, sep=';', encoding="utf-8-sig")
        b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">📥 Télécharger les résultats</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("✅ Prêt au téléchargement !")

def render_file_prediction():
    """Interface de prédiction par fichier"""
    st.markdown("## 📁 PRÉDICTION PAR FICHIER")
    st.markdown("Importez un fichier Excel contenant les données des étudiants :")
    
    # Zone de téléchargement
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Glissez-déposez votre fichier Excel ici",
        type=['xlsx', 'xls', 'csv'],
        help="Format accepté : .xlsx, .xls, .csv"
    )
    
    if uploaded_file:
        try:
            # Lire le fichier
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=';',encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file)
            
            # Afficher un aperçu
            st.success(f"✅ Fichier chargé : {uploaded_file.name}")
            st.write(f"**Nombre d'étudiants :** {len(df)}")
            
            # Aperçu des données
            with st.expander("👁️ Aperçu des données"):
                st.dataframe(df.head(), width=600)
            
            # Bouton de traitement
            if st.button("🎯 **PREDICTION**", key="process_file"):
                if not model_loaded:
                    st.error("⚠️ Modèle non chargé.")
                else:
                    with st.spinner("🔮 Traitement en cours..."):
                        try:
                            # Renommer les colonnes si nécessaire
                            column_mapping = {
                                'PHYSIQUE 2': 'Moyenne Pondéré PHYSIQUE 2',
                                'MATHÉMATIQUES 2': 'Moyenne Pondéré MATHÉMATIQUES 2',
                                'CHIMIE 2': 'Moyenne Pondéré CHIMIE 2',
                                'INFORMATIQUE 2': 'Moyenne Pondéré INFORMATIQUE 2',
                                'TRANSVERSALES 2': 'Moyenne Pondéré TRANSVERSALES 2',
                                'SCIENCES GRAPHIQUES': 'Moyenne Pondéré SCIENCES GRAPHIQUES'
                            }
                            
                            # Appliquer le mapping
                            df_for_pred = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                            
                            # Colonnes requises
                            required_columns = list(column_mapping.values())
                            
                            # Vérifier les colonnes
                            missing_cols = [col for col in required_columns if col not in df_for_pred.columns]
                            
                            if missing_cols:
                                st.error(f"Colonnes manquantes : {', '.join(missing_cols)}")
                            else:
                                # Faire les prédictions
                                predictions = model.predict(df_for_pred[required_columns])
                                probas = model.predict_proba(df_for_pred[required_columns])
                                
                                # Ajouter les résultats
                                results_df = df.copy()
                                results_df['PRÉDICTION'] = ['✅ VALIDÉ' if p == 1 else '❌ NON VALIDÉ' for p in predictions]
                                results_df['PROBABILITÉ'] = [p[1] * 100 for p in probas]
                                
                                # Calculer les statistiques
                                valid_count = sum(predictions)
                                total_count = len(predictions)
                                
                                # Afficher les résultats
                                col_res1, col_res2 = st.columns(2)
                                
                                with col_res1:
                                    st.markdown(f"""
                                    <div class="stats-box">
                                        <h3>✅ VALIDÉS</h3>
                                        <h1 style="font-size: 3rem;">{valid_count}</h1>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_res2:
                                    st.markdown(f"""
                                    <div class="stats-box" style="background: linear-gradient(135deg, #EF4444, #DC2626);">
                                        <h3>❌ NON VALIDÉS</h3>
                                        <h1 style="font-size: 3rem;">{total_count - valid_count}</h1>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Taux de réussite
                                taux = (valid_count / total_count * 100) if total_count > 0 else 0
                                st.markdown(f"""
                                <div class="stats-box" style="background: linear-gradient(135deg, #F59E0B, #D97706);">
                                    <h3>📈 TAUX DE RÉUSSITE</h3>
                                    <h1 style="font-size: 3rem;">{taux:.1f}%</h1>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Aperçu des résultats
                                with st.expander("📋 Voir les résultats détaillés"):
                                    st.dataframe(results_df, use_container_width=True)
                                
                                # Bouton d'export
                                csv = results_df.to_csv(index=False, sep=';',encoding="utf-8-sig")
                                b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="resultats_complets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">📥 Télécharger tous les résultats</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.error(f"❌ Erreur : {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Erreur de chargement : {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_about():
    """Section À propos simplifiée"""
    st.markdown("## ℹ️ À PROPOS")
    
    st.markdown("""
    <div class="input-card">
    <h3>🎓 PRÉDICTEUR ACADÉMIQUE</h3>
    
    **Version :** 1.0.0  
    **Développeur :** BARGO Alfred  
    
    ---
    
    **Description :**  
    Application de démonstration pour la prédiction des résultats de sessions académiques.
    
    **Fonctionnalités :**  
    • Prédiction manuelle  
    • Prédiction par fichier  
    • Export des résultats  
    
    ---
    
    *© 2026 - Application de démonstration*
    </div>
    """, unsafe_allow_html=True)

# ===========================================
# APPLICATION PRINCIPALE
# ===========================================
def main():
    # Charger le CSS personnalisé
    load_custom_css()
    
    # Afficher l'en-tête
    render_header()
    
    # Créer les onglets
    tab1, tab2, tab3 = st.tabs([
        "📝 PRÉDICTION MANUELLE", 
        "📁 PRÉDICTION PAR FICHIER", 
        "ℹ️ À PROPOS"
    ])
    
    # Onglet 1 : Prédiction Manuelle
    with tab1:
        render_manual_prediction()
    
    # Onglet 2 : Prédiction par Fichier
    with tab2:
        render_file_prediction()
    
    # Onglet 3 : À propos
    with tab3:
        render_about()
    
    # Footer
    st.markdown("""
    <div class="footer">
    <p>© 2026 - Prédicteur Académique - BARGO Alfred</p>
    <p style="font-size: 0.8rem;">Dernière mise à jour : """ + 
    datetime.now().strftime("%d/%m/%Y") + """</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Message d'information sur le modèle
    if not model_loaded:
        st.warning("""
        ⚠️ **Information :** Le modèle n'est pas chargé.  
        Assurez-vous que le fichier 'model.joblib' est présent dans le dossier courant.
        """)

if __name__ == "__main__":
    main()