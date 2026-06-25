"""
╔══════════════════════════════════════════════════════════════════════╗
║  CITADEL · Summer School 2025 · TP Déploiement — Phase 1           ║
║  Fichier : 02_serve.py                                              ║
║  Objectif : Exposer le modèle entraîné via une API REST (FastAPI)   ║
╚══════════════════════════════════════════════════════════════════════╝

Contexte
--------
Votre modèle est entraîné et sauvegardé dans model/sentiment/.
Maintenant il faut le rendre accessible : n'importe qui doit pouvoir
envoyer un texte et recevoir une prédiction, sans avoir Python installé.

La solution : une API REST. Un client envoie une requête HTTP avec
le texte en JSON, l'API retourne la prédiction en JSON.

Instructions
------------
Complétez les TODO dans l'ordre.
Prérequis : avoir exécuté 01_train.py avec succès.

Usage:
    uvicorn 02_serve:app --host 0.0.0.0 --port 8000 --reload
    Puis ouvrir : http://localhost:8000/docs
"""

import json
import pickle
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Chemin vers le modèle sauvegardé ─────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model" / "sentiment"

# ── État global — chargé UNE FOIS au démarrage ───────────────────────────────
# Pourquoi global ? Charger un modèle prend 100-200ms.
# Si on le chargeait à chaque requête, l'API serait 10x plus lente.
state = {}


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Charger le modèle au démarrage
# ══════════════════════════════════════════════════════════════════════════════

def charger_modele():
    """
    Charger le vectoriseur, le classifieur et les métadonnées depuis MODEL_DIR.

    Les trois fichiers à charger :
        model/sentiment/vectorizer.pkl  → avec pickle.load()
        model/sentiment/classifier.pkl  → avec pickle.load()
        model/sentiment/meta.json       → avec json.loads()

    Stockez tout dans le dictionnaire global `state` :
        state["vectoriseur"] = ...
        state["classifieur"] = ...
        state["classes"]     = meta["classes"]   # ["négatif", "positif"]
        state["meta"]        = meta
    """
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"Modèle introuvable dans {MODEL_DIR}/\n"
            "→ Lancez d'abord : python 01_train.py"
        )

    # Charger les métadonnées JSON
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Charger le vectoriseur avec pickle
    with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
        state["vectoriseur"] = pickle.load(f)

    # Charger le classifieur avec pickle
    with open(MODEL_DIR / "classifier.pkl", "rb") as f:
        state["classifieur"] = pickle.load(f)

    # Stocker les métadonnées dans state
    state["meta"]    = meta
    state["classes"] = meta["classes"]

    print(f"✓ Modèle chargé : {meta['type']} (accuracy: {meta['accuracy']:.1%})")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Fonction de prédiction
# ══════════════════════════════════════════════════════════════════════════════

def predire(texte: str) -> dict:
    """
    Prédire le sentiment d'un texte à partir du modèle chargé.

    Pipeline de prédiction :
        1. Vectoriser le texte avec state["vectoriseur"].transform([texte])
           ⚠️ .transform() et non .fit_transform() — on utilise le vocabulaire appris
        2. Prédire le label avec state["classifieur"].predict(X)
        3. Obtenir les probabilités avec .predict_proba(X)
        4. Construire et retourner le dictionnaire résultat

    Paramètres :
        texte : str — le texte à analyser

    Retourne :
        dict avec les clés :
            "label"   : str  — "positif" ou "négatif"
            "score"   : float — probabilité entre 0 et 1 (arrondie à 4 décimales)
            "positif" : bool  — True si positif
    """
    vec = state["vectoriseur"]
    clf = state["classifieur"]

    # Vectoriser le texte — liste d'UN seul élément, vocabulaire figé
    X = vec.transform([texte])

    # Prédire le label (0 ou 1)
    label_id = clf.predict(X)[0]

    # Obtenir la probabilité de la classe prédite
    # predict_proba retourne [[prob_négatif, prob_positif]]
    proba = clf.predict_proba(X)
    score = float(proba[0][label_id])

    # Récupérer le nom de la classe : state["classes"] = ["négatif", "positif"]
    label = state["classes"][label_id]

    return {
        "label":   label,
        "score":   round(score, 4),
        "positif": label == "positif",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Cycle de vie de l'application (chargement au démarrage)
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie FastAPI.
    Le code AVANT le `yield` s'exécute au démarrage.
    Le code APRÈS le `yield` s'exécute à l'arrêt.
    """
    print("\n" + "=" * 50)
    print("  CITADEL · Sentiment API — Démarrage")
    print("=" * 50)

    # Charger le modèle en mémoire une seule fois au démarrage
    charger_modele()

    print("\n  API prête sur http://0.0.0.0:8000")
    print("    → /docs  pour tester interactivement")
    print("    → /app   pour l'interface web\n")

    yield  # L'application tourne ici

    print("API arrêtée.")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Définir l'application et les endpoints
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="CITADEL · Sentiment API",
    description="Classificateur de sentiment en français — TP Déploiement Phase 1",
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_parameters={"syntaxHighlight": False},
)

# Autoriser les appels depuis le navigateur (interface web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schémas de données ────────────────────────────────────────────────────────

class TexteEntree(BaseModel):
    """Corps de la requête POST /predict"""
    texte: str   # Le texte à analyser


class PredictionSortie(BaseModel):
    """Corps de la réponse"""
    texte:   str
    label:   str
    score:   float
    positif: bool
    modele:  str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionSortie)
def predire_sentiment(corps: TexteEntree):
    """
    Analyser le sentiment d'un texte en français.
    """
    # Valider l'entrée : refuser un texte vide ou uniquement des espaces
    if not corps.texte.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    # Appeler predire() et récupérer le résultat
    resultat = predire(corps.texte)

    # Retourner la réponse — **resultat dépaquète label, score, positif
    return PredictionSortie(
        texte=corps.texte,
        modele=state["meta"]["type"],
        **resultat,
    )


@app.get("/health")
def etat():
    """
    Retourner l'état du service.
    """
    return {
        "status":   "ok",
        "modele":   state["meta"]["type"],
        "accuracy": state["meta"]["accuracy"],
        "classes":  state["classes"],
    }


@app.get("/app", response_class=HTMLResponse)
def interface():
    """Servir l'interface web."""
    html = Path(__file__).parent / "app" / "index.html"
    if html.exists():
        return HTMLResponse(content=html.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h3>Interface introuvable — créez app/index.html (Étape bonus)</h3>",
        status_code=404
    )


@app.get("/")
def accueil():
    """Page d'accueil avec la liste des endpoints."""
    return {
        "service": "CITADEL Sentiment API",
        "endpoints": {
            "POST /predict": "Analyser un texte",
            "GET  /health" : "État du modèle en production",
            "GET  /app"    : "Interface web",
            "GET  /docs"   : "Documentation interactive Swagger",
        },
    }
    
