
# 🌲 Prédiction des Régimes de Perturbations en Forêt Boréale par Apprentissage Automatique

> **Projet de Recherche & Modélisation Hybride (IA & Systèmes Dynamiques)**
> *Développement de modèles d'apprentissage profond pour séries temporelles paléoécologiques (12 000 ans BP) afin de prédire la fréquence et l'intensité des feux de forêt boréaux.*

---

## 📌 Présentation du Projet

Ce projet s'inscrit dans le cadre d'une étude sur la dynamique long terme des forêts boréales face au changement climatique. Il vise à lever un verrou scientifique majeur : **capturer la stochastique et la non-linéarité des perturbations environnementales (incendies de forêt)** pour alimenter un modèle mathématique sous-jacent d'équations différentielles régissant la biomasse forestière.

À partir d'un jeu de données paléoécologiques reconstruisant 12 000 ans d'histoire écologique au Québec, nous entraînons et évaluons des architectures de **Machine Learning classique** (Random Forest, Gradient Boosting) et de **Deep Learning séquentiel** (N-BEATS, LSTMPatch, TransformerPatch, Temporal Transformer).

### 🎯 Principaux Résultats

* **Fréquence du feu ($\lambda$ / RegFRI) :** L'architecture **N-BEATS** s'impose sur les horizons étendus ($192 \to 48$ pas) avec un coefficient de détermination exceptionnel ($R^2 = 0{,}9994$, $\text{RMSE} = 2{,}2441$).
* **Intensité du feu ($\gamma$ / RegFS) :** Le modèle **TransformerPatch** excelle à capturer la variabilité locale et la bimodalité du signal au premier pas ($R^2 = 0{,}9468$, $\text{RMSE} = 0{,}0124$ à $h+1$).

---

## 🗂️ Structure du Dépôt

```bash
.
├── figures_feu/                # Graphiques descriptifs du signal régional et des occurrences lacustres
├── figures_notebook_outputs/    # Graphiques de sorties expérimentales (prédictions vs obs, résidus, dérives)
├── model_weights/              # Poids exportés et pipelines sérialisés
│   ├── frequency/              # Modèles entraînés sur la fréquence du feu (RegFRI / FF)
│   │   ├── proxy_frequency_NBeats.pth
│   │   ├── proxy_frequency_TransformerPatch.pth
│   │   ├── proxy_frequency_LSTMPatch.pth
│   │   ├── proxy_frequency_TemporalFusionTransformer.pth
│   │   ├── proxy_frequency_RandomForest.joblib
│   │   └── ...
│   └── intensity/              # Modèles entraînés sur l'intensité du feu (RegFS / FS)
│       ├── proxy_intensity_TransformerPatch.pth
│       ├── proxy_intensity_NBeats.pth
│       ├── proxy_intensity_RandomForest.joblib
│       └── ...
│
├── Analyse_exploratoire_donnees.ipynb        # Exploration descriptive & analyse spatio-temporelle des proxys
├── exploration_variables_modeles_feu.ipynb   # Pipeline d'entraînement, validation séquentielle & benchmark DL
├── plus_loin*.ipynb                          # Expérimentations avancées, tests d'horizon & stabilité auto-régressive
│
├── *.csv                       # Tables de données de travail (séries temporelles régionales et lacustres)
├── requirements.txt            # Liste des dépendances Python
└── README.md                   # Documentation du projet
```

---

## 💾 Utilisation des Modèles Entraînés (`model_weights/`)

Les poids et pipelines pré-entraînés sont sauvegardés dans le dossier `model_weights/` afin d'effectuer des inférences directes sans réentraîner les réseaux.

### 1. Charger un modèle classique (Scikit-Learn / `.joblib`)

Les modèles classiques intègrent directement leur pipeline de prétraitement :

```python
import joblib

# Chargement du RandomForest entraîné sur la fréquence
rf_model = joblib.load('model_weights/frequency/proxy_frequency_RandomForest.joblib')

# Prédiction
predictions = rf_model.predict(X_test)

```

### 2. Charger un modèle séquentiel (PyTorch / `.pth`)

Pour charger les réseaux de neurones profonds, instanciez l'architecture correspondante avec sa configuration avant d'injecter les poids :

```python
import torch
from your_module import NBeatsNet  # Remplacer par la classe de votre modèle

# 1. Instanciation de l'architecture avec la configuration adéquate
model = NBeatsNet(input_dim=24, horizon=6)

# 2. Chargement des poids enregistrés
weights_path = 'model_weights/frequency/proxy_frequency_NBeats.pth'
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# 3. Passage en mode évaluation
model.eval()

# 4. Inférence sur un tenseur d'entrée (batch_size, seq_len)
with torch.no_grad():
    predictions = model(input_tensor)

```

---

## 📊 Données et Sources Scientifiques

Les données utilisées combinent des enregistrements de micro-charbons et de pollen issus de sédiments lacustres au Québec sur les 12 000 dernières années :

* **Proxys régionaux (`*BootCISap.csv`) :** Fréquence régionale ($\text{FF}$), Biomasse brûlée ($\text{BB}$) et Indice de sévérité ($\text{FS} = \text{BB}/\text{FF}$).
* **Occurrences locales (`Lake_fire_frequency_*.csv`) :** 297 feux régionaux représentés par 375 détections locales réparties sur 6 sites lacustres (M15, O15, O14, O16, O6, M14).

### 📜 Référence Bibliographique

La source primaire des données provient de la publication de référence :

> Girardin, M. P., Gaboriau, D. M., Ali, A. A., Gajewski, K., Briere, M. D., Bergeron, Y., Paillard, J., Waito, J., & Tardif, J. C. (2024). *Boreal forest cover was reduced in the mid-Holocene with warming and recurring wildfires*. **Communications Earth & Environment**, 5(1), 176. [https://doi.org/10.1038/s43247-024-01340-8](https://doi.org/10.1038/s43247-024-01340-8?utm_source=gemini)

---

## 🛠️ Installation et Environnement

### Préréquis

* Python 3.9+
* PyTorch (avec support CUDA recommandé si disponible)

### Installation

1. Cloner le dépôt localement :

```bash
git clone [https://github.com/richUlric/Stage_Recherche_Prediction_Feu_foret.git](https://github.com/richUlric/Stage_Recherche_Prediction_Feu_foret.git)
cd Stage_Recherche_Prediction_Feu_foret

```

2. Créer et activer un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/macOS
# venv\Scripts\activate   # Sur Windows
```

---

## 🚀 Reproduction des Analyses

Pour reproduire l'ensemble des figures et des tableaux de performances présentés dans les rapports :

1. S'assurer que les fichiers de données `.csv` sont bien présents à la racine du dépôt.
2. Lancer un environnement Jupyter (ou VS Code) :

```bash
jupyter lab

```

3. Exécuter les notebooks dans l'ordre préconisé :

* **`Analyse_exploratoire_donnees.ipynb`** : Génération des cartes de chaleur, distributions d'âges et historiques régionaux (dossier `figures_feu/`).
* **`exploration_variables_modeles_feu.ipynb`** : Prétraitement (windowing, transformation empirique $1/x$), entraînement des modèles séquentiels PyTorch et génération des métriques à $h+1$ et multi-horizons.
* **`plus_loin*.ipynb`** : Test d'auto-régression longue durée et analyse des seuils de dérive temporelle (ex. limite de stabilité à 162 pas).

---

## 🤝 Encadrement & Institutions

Ce projet s'inscrit dans le cadre d'un stage de fin d'études mené en collaboration avec :

* **Laboratoire :** Laboratoire des Sciences du Numérique de Nantes (LS2N - UMR CNRS 6004), Équipe VELO.
* **Établissements :** Université de Nantes & École Centrale de Nantes.

```

```
