"""Perception du contrôle à la voix : le préparateur décrit l'article à l'oral, une reconnaissance vocale (STT) le transcrit.

La voix du préparateur est synthétisée (macOS), pour que la mesure soit reproductible ; la transcription passe par un
modèle qui accepte l'audio, via OpenRouter. On mesure la latence de la transcription et on la compare au texte prononcé.
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import statistics
import subprocess
import tempfile
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf

from reachy_jev import openrouter
from reachy_jev.reglages import DOSSIER_RESULTATS, MODELE_STT, RACINE, VOIX_PREPARATEUR

FICHIER = DOSSIER_RESULTATS / "stt.json"
DOSSIER_AUDIO = RACINE / ".cache" / "preparateur"
SEUIL_EXACTE = 0.95  # similarité à partir de laquelle une transcription compte comme exacte
CONSIGNE = "Transcris exactement ce que dit cette personne, en français, sans rien ajouter ni commenter."


def enregistrer_phrase(texte: str, dossier: Path = DOSSIER_AUDIO) -> Path:
    """La phrase du préparateur en WAV 16 kHz mono (voix macOS), mise en cache."""
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{hashlib.sha256(f'{VOIX_PREPARATEUR}|{texte}'.encode()).hexdigest()[:24]}.wav"
    if not chemin.exists():
        with tempfile.TemporaryDirectory() as tmp:
            provisoire = Path(tmp) / "phrase.wav"
            subprocess.run(
                ["say", "-v", VOIX_PREPARATEUR, "--file-format=WAVE", "--data-format=LEI16@16000", "-o", str(provisoire), texte],
                check=True,
                timeout=30,
            )
            provisoire.replace(chemin)
    return chemin


def transcrire(client: httpx.Client, chemin_wav: Path) -> tuple[str | None, openrouter.Appel]:
    """La transcription d'un WAV, et l'appel (latence, coût). Le texte vaut None en cas d'échec."""
    audio = base64.b64encode(chemin_wav.read_bytes()).decode()
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": CONSIGNE}, {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}],
        }
    ]
    appel = openrouter.chat(client, {"model": MODELE_STT, "temperature": 0}, messages, max_tokens=120)
    texte = openrouter.texte(appel)
    return (texte.strip() if texte else None), appel


def _normaliser(texte: str) -> str:
    """Minuscules, sans accents ni ponctuation : on compare les mots, pas la typographie."""
    sans_accents = unicodedata.normalize("NFKD", texte.lower()).encode("ascii", "ignore").decode()
    return " ".join("".join(c if c.isalnum() else " " for c in sans_accents).split())


def similarite(attendu: str, obtenu: str | None) -> float:
    return difflib.SequenceMatcher(None, _normaliser(attendu), _normaliser(obtenu or "")).ratio()


def mesurer(client: httpx.Client, phrases: list[str]) -> dict[str, Any]:
    """Chaque phrase : synthèse (hors mesure), puis transcription chronométrée. Un appel de chauffe non compté."""
    audios = {phrase: enregistrer_phrase(phrase) for phrase in phrases}
    transcrire(client, audios[phrases[0]])  # chauffe
    lignes = []
    for phrase, chemin in audios.items():
        transcription, appel = transcrire(client, chemin)
        lignes.append(
            {
                "texte": phrase,
                "transcription": transcription,
                "similarite": round(similarite(phrase, transcription), 3),
                "duree_audio_s": round(sf.info(str(chemin)).duration, 2),
                "latence_ms": round(appel.latence_ms),
                "cout_usd": appel.cout_usd,
                "erreur": appel.erreur,
            }
        )
    valides = [ligne for ligne in lignes if not ligne["erreur"]]
    return {
        "date": date.today().isoformat(),
        "modele": MODELE_STT,
        "voix": f"macOS « {VOIX_PREPARATEUR} »",
        "phrases": lignes,
        "latence_mediane_ms": round(statistics.median(ligne["latence_ms"] for ligne in valides)) if valides else None,
        "duree_audio_mediane_s": round(statistics.median(ligne["duree_audio_s"] for ligne in lignes), 2),
        "transcriptions_exactes": sum(ligne["similarite"] >= SEUIL_EXACTE for ligne in valides),
        "cout_moyen_usd": statistics.fmean(ligne["cout_usd"] for ligne in valides if ligne["cout_usd"] is not None) if valides else None,
    }


def ecrire(mesure: dict[str, Any], chemin: Path = FICHIER) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(mesure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return chemin


def lire(chemin: Path = FICHIER) -> dict[str, Any] | None:
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else None
