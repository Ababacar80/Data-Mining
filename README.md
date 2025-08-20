# **Prédiction de Remboursement de Prêt avec un Réseau de Neurones Artificiels (ANN)**

## **Table des matières**

1.  [Description du Projet]
2.  [Objectif]
3.  [Les Données]
4.  [Démarche du Projet]
      - [1. Analyse Exploratoire des Données (EDA)]
      - [2. Prétraitement et Ingénierie des Caractéristiques]
      - [3. Construction du Modèle de Deep Learning]
      - [4. Entraînement et Évaluation du Modèle]
5.  [Résultats]
6.  [Installation et Dépendances]
7.  [Comment l'utiliser]

-----

## **Description du Projet**

Ce projet met en œuvre un modèle d'apprentissage profond pour aborder un problème de classification binaire dans le secteur financier. En utilisant un ensemble de données historiques de **LendingClub**, la plus grande plateforme de prêt entre pairs au monde, nous construisons un réseau de neurones artificiels (ANN) capable de prédire la probabilité qu'un emprunteur rembourse intégralement son prêt.

L'analyse couvre l'ensemble du pipeline de la science des données, de l'exploration et du nettoyage des données à la construction, l'entraînement et l'évaluation du modèle.

-----

## **Objectif**

L'objectif principal est de fournir à LendingClub un outil d'aide à la décision fiable. Le modèle doit évaluer les nouveaux demandeurs de prêt et prédire leur capacité de remboursement. Cela permet de minimiser le risque de défaut de paiement (charge-off) et d'optimiser l'octroi de crédits. La colonne cible pour la prédiction est `loan_status`, transformée en une variable binaire : `1` pour "Fully Paid" (Remboursé) et `0` pour "Charged Off" (Défaut de paiement).

-----

## **Les Données**

Le projet utilise le fichier `lending_club_loan_two.csv`. Ce jeu de données contient des informations détaillées sur chaque prêt, y compris :

  * **Informations sur le prêt :** `loan_amnt` (montant), `term` (durée), `int_rate` (taux d'intérêt), `grade` (notation du prêt).
  * **Informations sur l'emprunteur :** `emp_title` (profession), `home_ownership` (statut du logement), `annual_inc` (revenu annuel), `verification_status`.
  * **Historique de crédit :** `dti` (ratio dette/revenu), `earliest_cr_line` (date de la première ligne de crédit), `revol_util` (utilisation du crédit renouvelable), `total_acc` (nombre total de comptes de crédit).
  * **Variable cible :** `loan_status` (statut final du prêt).

-----

## **Démarche du Projet**

### **1. Analyse Exploratoire des Données (EDA)**

La première phase consiste à comprendre les données en profondeur à travers des visualisations pour identifier les tendances, les corrélations et les anomalies.

  * **Distribution de la variable cible :** Un graphique à barres (`countplot`) montre un déséquilibre entre les prêts remboursés et ceux en défaut de paiement.
  * **Analyse des variables numériques :**
      * Des histogrammes sont utilisés pour visualiser la distribution de variables clés comme `loan_amnt`, `int_rate`, et `annual_inc`.
      * Une **matrice de corrélation** (`heatmap`) est générée pour identifier les relations linéaires entre les variables. Une forte corrélation est observée entre `loan_amnt` et `installment`.
  * **Analyse des variables catégorielles :** Des graphiques à barres explorent la relation entre des caractéristiques comme `grade`, `sub_grade`, `home_ownership` et le statut du prêt.

### **2. Prétraitement et Ingénierie des Caractéristiques**

Cette étape cruciale prépare les données pour l'entraînement du modèle.

  * **Gestion des valeurs manquantes :** Analyse du pourcentage de valeurs manquantes pour chaque colonne. Les colonnes avec un grand nombre de valeurs manquantes (comme `emp_title`) sont supprimées, tandis que d'autres (comme `revol_util` et `mort_acc`) sont imputées en utilisant des stratégies appropriées (par exemple, la moyenne).
  * **Transformation des caractéristiques :**
      * **Variables catégorielles :** Conversion des variables textuelles en variables numériques via le "one-hot encoding" (`pd.get_dummies`). Cela inclut des colonnes comme `verification_status`, `application_type`, `initial_list_status`, `purpose`, etc. Les colonnes `grade` et `sub_grade` sont également transformées.
      * **Caractéristiques de date :** La colonne `earliest_cr_line` est convertie en une caractéristique numérique représentant l'année.
      * **Caractéristiques textuelles :** La colonne `term` est convertie en une valeur numérique (36 ou 60 mois).
  * **Mise à l'échelle des données :** Toutes les caractéristiques numériques sont normalisées à l'aide de `MinMaxScaler` pour garantir qu'elles se situent dans une plage similaire (entre 0 et 1), ce qui est essentiel pour la performance des réseaux de neurones.

### **3. Construction du Modèle de Deep Learning**

Un réseau de neurones séquentiel est construit à l'aide de la bibliothèque **Keras** (avec TensorFlow en backend).

  * **Architecture du modèle :**
      * **Couche d'entrée :** Une couche `Dense` avec 78 neurones (correspondant au nombre de caractéristiques d'entrée) et une fonction d'activation **ReLU**.
      * **Couches cachées :** Deux couches `Dense` cachées avec respectivement 39 et 19 neurones, utilisant également l'activation **ReLU**.
      * **Couche de sortie :** Une couche `Dense` finale avec un seul neurone et une fonction d'activation **sigmoïde**, car il s'agit d'un problème de classification binaire.
  * **Compilation du modèle :**
      * **Fonction de perte :** `binary_crossentropy` est choisie, car elle est adaptée à la classification binaire.
      * **Optimiseur :** L'optimiseur `adam` est utilisé pour ajuster les poids du réseau de manière efficace.
      * **Métriques :** `accuracy` est suivie pendant l'entraînement.

### **4. Entraînement et Évaluation du Modèle**

  * **Division des données :** Le jeu de données est divisé en un ensemble d'entraînement (80%) et un ensemble de test (20%) en utilisant `train_test_split`.
  * **Entraînement :** Le modèle est entraîné sur les données d'entraînement pendant 25 époques avec une taille de batch de 256. Des techniques comme le `EarlyStopping` sont utilisées pour prévenir le surapprentissage en surveillant la perte de validation.
  * **Évaluation :** Les performances du modèle sont évaluées sur l'ensemble de test à l'aide de :
      * **Rapport de classification :** Affiche la **précision**, le **rappel** et le **F1-score** pour chaque classe.
      * **Matrice de confusion :** Visualise les vrais positifs, les faux positifs, les vrais négatifs et les faux négatifs.
      * **Perte et Précision :** Les courbes de perte et de précision de l'entraînement et de la validation sont tracées pour visualiser la convergence du modèle.

-----

## **Résultats**

Le modèle atteint une **précision d'environ 89%** sur l'ensemble de test. Le rapport de classification détaillé et la matrice de confusion montrent que le modèle est performant, en particulier pour prédire les prêts qui seront entièrement remboursés. Le notebook se conclut par une démonstration de la manière d'utiliser le modèle entraîné pour prédire le statut d'un nouveau client aléatoire.

-----

## **Installation et Dépendances**

Pour exécuter ce projet, vous aurez besoin des bibliothèques Python suivantes :

  * pandas
  * numpy
  * matplotlib
  * seaborn
  * scikit-learn
  * tensorflow

Vous pouvez installer toutes les dépendances nécessaires à l'aide de `pip` :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

-----

## **Comment l'utiliser**

1.  Clonez ce dépôt sur votre machine locale.
2.  Assurez-vous que le fichier de données `lending_club_loan_two.csv` est placé dans le même répertoire que le notebook.
3.  Lancez un environnement Jupyter (Jupyter Notebook ou JupyterLab).
4.  Ouvrez et exécutez le notebook `Projet_ANN_Ababacar_Sagna.ipynb` cellule par cellule.
