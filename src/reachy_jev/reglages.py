"""Réglages : clé (fichier .env), cerveaux comparés, seuils, adresse du robot."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RACINE = Path(__file__).resolve().parents[2]
load_dotenv(RACINE / ".env")  # n'écrase pas une variable déjà définie dans l'environnement

# OPENROUTER_URL ne sert qu'à pointer vers un faux serveur pendant les tests.
URL_OPENROUTER = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api").rstrip("/")
URL_DECISIONS = f"{URL_OPENROUTER}/alpha/decisions"
URL_CHAT = f"{URL_OPENROUTER}/v1/chat/completions"

# Les trois cerveaux comparés, tous derrière la même clé OpenRouter :
# - deux modèles de DÉCISION (même API, seul `model` change) ;
# - le LLM GÉNÉRATIF de la même famille qu'OpenAI Decisions, dans son réglage le plus rapide, pour mesurer ce que coûte la génération.
# Versions épinglées : un seuil de confiance réglé sur une version ne vaut que pour elle.
CERVEAUX: dict[str, dict] = {
    "Jev 1.14": {"type": "decision", "model": "typesafe/jev-1.14"},
    "GPT-6 Luna Decisions": {"type": "decision", "model": "openai/gpt-6-luna-decisions"},
    "GPT-6 Luna (génératif)": {"type": "llm", "model": "openai/gpt-6-luna", "reasoning": {"effort": "minimal"}},
}
CERVEAU_PILOTE = "Jev 1.14"  # celui qui fait bouger le robot

# Reconnaissance vocale (STT) du contrôle à la voix : un modèle qui accepte l'audio, via la même clé OpenRouter.
MODELE_STT = "google/gemini-2.5-flash-lite"
VOIX_PREPARATEUR = "Jacques"  # voix macOS de synthèse du préparateur, distincte de celle du robot (« Thomas »)

# Zone de doute sur « est-ce le bon produit ? » (probabilité du oui) : entre les deux, le robot demande de remontrer l'article.
DOUTE_BAS, DOUTE_HAUT = 0.3, 0.7
SEUIL_CONFIANCE_ETAT = 0.5  # plancher conseillé par la doc TypeSafe

# Toutes les photos sont ramenées à ce côté maximal, et envoyées identiques aux trois cerveaux (Jev accepte 256 à 1 568 px).
COTE_MAX_IMAGE = 1024

DOSSIER_DONNEES = RACINE / "data"
DOSSIER_RESULTATS = RACINE / "resultats"

MSG_CLE_MANQUANTE = (
    "OPENROUTER_API_KEY absente. Crée une clé sur https://openrouter.ai/settings/keys, puis copie .env.example en .env et colle-la dedans."
)


def cle_openrouter() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY", "").strip() or None


def hote_robot() -> str:
    return os.environ.get("REACHY_HOTE", "").strip() or "reachy-mini.local"
