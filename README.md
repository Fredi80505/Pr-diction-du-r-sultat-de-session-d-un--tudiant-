# 🎓 Prédiction des Résultats de Session des Étudiants

Ce projet vise à prédire le résultat de fin de session d’un étudiant (**Validé** / **Non validé**) à partir de ses moyennes pondérées dans six matières.  
Il se compose :

- d’un **notebook Jupyter** pour l’analyse, le prétraitement et l’entraînement du modèle
- d’une **application interactive** construite avec **Streamlit**

---

## 📁 Contenu du projet


├── Data_training.xlsx # Données brutes

├── Prédiction du résultat de session des étudiants.ipynb # Notebook d'analyse et de modélisation

├── model.joblib # Modèle entraîné (généré par le notebook)

├── app_streamlit_simple.py # Application Streamlit

├── assets/ # Dossier des images utilisées par l'application

│ ├── background.png

│ ├── valide.jpg

│ └── non_valide.jpg

└── README.md



---

## 📊 Données

- 1087 étudiants, 6 notes pondérées + 1 colonne « Total » (retirée car redondante)
- Variable cible : `Résultat` (Validé / Non validé)
- Déséquilibre important : ~80% de « Non validé »

---

## 🛠️ Technologies utilisées

- Python 3
- pandas, numpy, matplotlib, seaborn, scipy
- imbalanced-learn (SMOTE)
- scikit‑learn (RandomForest, métriques, validation croisée)
- joblib (sauvegarde du modèle)
- Streamlit (interface utilisateur)

---

## ⚙️ Installation

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn joblib streamlit pillow openpyxl


## 🖥️ Application Streamlit
L’application permet de prédire le résultat d’un étudiant de deux manières :

- Saisie manuelle des notes (6 champs)
- Import d’un fichier Excel/CSV contenant plusieurs étudiants

### Lancement
- Vérifie que le fichier model.joblib est présent dans le dossier du projet
- Place les images dans le dossier assets/
- Exécute : streamlit run app_streamlit_simple.py

### Fonctionnalités de l’application
- Affichage du résultat (Validé/Non validé) avec probabilité
- Statistiques globales pour les traitements par lot
- Export des résultats au format CSV


👤 Auteur
BARGO Alfred
