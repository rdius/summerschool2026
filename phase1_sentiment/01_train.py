"""
╔══════════════════════════════════════════════════════════════════════╗
║  CITADEL · Summer School 2025 · TP Déploiement — Phase 1           ║
║  Fichier : 01_train.py                                              ║
║  Objectif : Entraîner un classificateur de sentiment et sauvegarder ║
║             les artefacts nécessaires au déploiement                ║
╚══════════════════════════════════════════════════════════════════════╝

Contexte
--------
Un service reçoit des milliers d'avis clients en français.
Votre mission : entraîner un modèle capable de classer chaque avis
en POSITIF (1) ou NÉGATIF (0), puis le sauvegarder pour le déployer.

Instructions
------------
Cherchez les blocs marqués TODO et complétez-les.
Les commentaires vous guident étape par étape.
Ne modifiez pas les parties hors TODO.

Usage:
    python 01_train.py
"""

import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ── Répertoire de sauvegarde du modèle ───────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model" / "sentiment"

# ── Dataset : avis en français avec leurs labels ──────────────────────────────
# 1 = positif, 0 = négatif
TRAIN_DATA = [
    ("Ce service est excellent, je suis très satisfait !", 1),
    ("Produit de qualité, livraison rapide, je recommande.", 1),
    ("Très bonne expérience, le personnel est accueillant.", 1),
    ("Parfait, conforme à la description, rien à redire.", 1),
    ("Incroyable qualité pour ce prix, je suis bluffé.", 1),
    ("Super rapport qualité/prix, je suis ravi de mon achat.", 1),
    ("Excellent produit, fonctionne parfaitement depuis 3 mois.", 1),
    ("Très déçu, le produit ne correspond pas du tout à la description.", 0),
    ("Mauvaise qualité, tombé en panne après une semaine.", 0),
    ("Service client inexistant, aucune réponse à mes mails.", 0),
    ("Livraison en retard, emballage abîmé, produit défectueux.", 0),
    ("Arnaque totale, je déconseille fortement ce vendeur.", 0),
    ("Très mauvaise expérience, je ne recommande pas du tout.", 0),
    ("Produit inutilisable, retour impossible, argent perdu.", 0),
    ("Satisfait de ma commande, arrivée dans les délais.", 1),
    ("Déçu par la qualité, je m'attendais à mieux.", 0),
    ("Excellent, exactement ce que je cherchais !", 1),
    ("Pas terrible, qualité médiocre pour le prix demandé.", 0),
    ("Très bien, je suis content de mon achat.", 1),
    ("Horrible expérience, à éviter absolument.", 0),
]

TEST_DATA = [
    ("Vraiment bien, je suis content de cet achat.", 1),
    ("Nul, ne fonctionne pas, à éviter.", 0),
    ("Parfait pour mon usage, je recommande.", 1),
    ("Mauvaise qualité, je suis déçu.", 0),
    ("Super produit, livraison rapide !", 1),
    ("Pas convaincu, qualité insuffisante.", 0),
]


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Préparer les données
# ══════════════════════════════════════════════════════════════════════════════

def preparer_donnees(dataset):
    """
    Séparer les textes et les labels depuis une liste de tuples (texte, label).

    Exemple :
        dataset = [("Très bien !", 1), ("Mauvais produit", 0)]
        → textes = ["Très bien !", "Mauvais produit"]
        → labels = [1, 0]

    Paramètres :
        dataset : list of (str, int)

    Retourne :
        textes : list[str]
        labels : list[int]
    """
    textes = [texte for texte, label in dataset]
    labels = [label for texte, label in dataset]

    return textes, labels


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Vectoriser le texte
# ══════════════════════════════════════════════════════════════════════════════

def creer_vectoriseur():
    """
    Créer un TfidfVectorizer configuré pour le français.

    TF-IDF transforme un texte en vecteur numérique :
    chaque mot fréquent dans ce texte mais rare dans les autres
    reçoit un score élevé.

    Paramètres recommandés :
        - ngram_range=(1, 2)  : prendre les mots seuls ET les bigrammes
                                ex : "pas bien" est différent de "bien"
        - max_features=5000   : garder les 5000 features les plus utiles

    Retourne :
        TfidfVectorizer
    """
    vectoriseur = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

    return vectoriseur


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraîner le classifieur
# ══════════════════════════════════════════════════════════════════════════════

def entrainer(vectoriseur, textes_train, labels_train):
    """
    Vectoriser les textes d'entraînement et entraîner un classifieur.

    Deux sous-étapes :
        1. vectoriseur.fit_transform(textes_train)
           → apprend le vocabulaire ET transforme les textes en matrice
        2. Créer et entraîner un LogisticRegression

    Paramètres :
        vectoriseur    : TfidfVectorizer (non encore entraîné)
        textes_train   : list[str]
        labels_train   : list[int]

    Retourne :
        classifieur : LogisticRegression entraîné
        X_train     : matrice TF-IDF des données d'entraînement
    """
    # fit_transform : apprend le vocabulaire ET transforme en une seule passe
    X_train = vectoriseur.fit_transform(textes_train)

    classifieur = LogisticRegression(max_iter=1000)
    classifieur.fit(X_train, labels_train)

    return classifieur, X_train


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Évaluer le modèle
# ══════════════════════════════════════════════════════════════════════════════

def evaluer(vectoriseur, classifieur, textes_test, labels_test):
    """
    Évaluer le modèle sur les données de test.

    Sous-étapes :
        1. Vectoriser les textes de test avec vectoriseur.transform()
           (PAS fit_transform — on ne réapprend pas le vocabulaire sur le test !)
        2. Prédire avec classifieur.predict()
        3. Afficher le rapport et retourner l'accuracy

    Paramètres :
        vectoriseur  : TfidfVectorizer déjà entraîné
        classifieur  : LogisticRegression déjà entraîné
        textes_test  : list[str]
        labels_test  : list[int]

    Retourne :
        accuracy : float entre 0 et 1
    """
    # .transform() uniquement : on applique le vocabulaire appris sur le train
    X_test = vectoriseur.transform(textes_test)

    predictions = classifieur.predict(X_test)

    print("\n=== Rapport d'évaluation ===")
    print(classification_report(labels_test, predictions, target_names=["négatif", "positif"]))

    accuracy = accuracy_score(labels_test, predictions)

    return accuracy


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Sauvegarder les artefacts de déploiement
# ══════════════════════════════════════════════════════════════════════════════

def sauvegarder(vectoriseur, classifieur, accuracy):
    """
    Sauvegarder les artefacts nécessaires au déploiement.

    Trois fichiers à créer dans MODEL_DIR :
        vectorizer.pkl  → le TfidfVectorizer entraîné (sérialisation pickle)
        classifier.pkl  → le LogisticRegression entraîné (sérialisation pickle)
        meta.json       → métadonnées : type, classes, accuracy

    La sérialisation pickle convertit un objet Python en bytes
    qu'on peut stocker sur disque et recharger plus tard.

    Paramètres :
        vectoriseur : TfidfVectorizer entraîné
        classifieur : LogisticRegression entraîné
        accuracy    : float
    """
    # Créer le répertoire si nécessaire
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectoriseur, f)

    with open(MODEL_DIR / "classifier.pkl", "wb") as f:
        pickle.dump(classifieur, f)

    meta = {
        "type": "tfidf+logreg",
        "classes": ["négatif", "positif"],
        "accuracy": accuracy,
    }
    with open(MODEL_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Modèle sauvegardé dans {MODEL_DIR}/")
    print(f"  • vectorizer.pkl")
    print(f"  • classifier.pkl")
    print(f"  • meta.json")
    print(f"\n  Accuracy : {accuracy:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL — Ne pas modifier
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  CITADEL · Entraînement classificateur de sentiment")
    print("=" * 60)

    # 1. Préparer les données
    print("\n[1/5] Préparation des données...")
    textes_train, labels_train = preparer_donnees(TRAIN_DATA)
    textes_test,  labels_test  = preparer_donnees(TEST_DATA)
    print(f"      Train : {len(textes_train)} exemples")
    print(f"      Test  : {len(textes_test)} exemples")

    # 2. Créer le vectoriseur
    print("\n[2/5] Création du vectoriseur TF-IDF...")
    vectoriseur = creer_vectoriseur()

    # 3. Entraîner
    print("\n[3/5] Entraînement du classifieur...")
    classifieur, X_train = entrainer(vectoriseur, textes_train, labels_train)
    print(f"      Matrice d'entraînement : {X_train.shape}")

    # 4. Évaluer
    print("\n[4/5] Évaluation sur les données de test...")
    accuracy = evaluer(vectoriseur, classifieur, textes_test, labels_test)

    # 5. Sauvegarder
    print("\n[5/5] Sauvegarde des artefacts de déploiement...")
    sauvegarder(vectoriseur, classifieur, accuracy)

    print("\n  Entraînement terminé ! Passez à 02_serve.py")
