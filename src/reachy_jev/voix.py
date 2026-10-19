"""Les phrases du robot, synthétisées une fois au lancement puis relues depuis le cache : la parole part sans aller-retour réseau.

ElevenLabs si ELEVENLABS_API_KEY et ELEVENLABS_VOICE_ID sont dans .env ; sinon la voix française macOS « Thomas », hors ligne.
Ne lève jamais : une voix en panne laisse le robot muet, elle ne l'arrête pas.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from reachy_jev.reglages import RACINE

logger = logging.getLogger(__name__)

DOSSIER_VOIX = RACINE / ".cache" / "voix"
URL_ELEVENLABS = "https://api.elevenlabs.io/v1/text-to-speech/{voix_id}"


def moteur() -> str:
    voix_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    return f"elevenlabs:{voix_id}" if os.environ.get("ELEVENLABS_API_KEY", "").strip() and voix_id else "macos:Thomas"


def _elevenlabs(texte: str, voix_id: str, chemin: Path) -> bool:
    try:
        reponse = httpx.post(
            URL_ELEVENLABS.format(voix_id=voix_id),
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            params={"output_format": "pcm_24000"},
            json={"text": texte, "model_id": "eleven_multilingual_v2"},
            timeout=20,
        )
        reponse.raise_for_status()
    except Exception as exc:
        logger.warning("ElevenLabs indisponible (%s), voix macOS.", exc)
        return False
    sf.write(str(chemin), np.frombuffer(reponse.content, dtype="<i2"), 24000, subtype="PCM_16")
    return True


def _macos(texte: str, chemin: Path) -> bool:
    with tempfile.TemporaryDirectory() as dossier:
        aiff = Path(dossier) / "parole.aiff"
        try:
            subprocess.run(["say", "-v", "Thomas", "-o", str(aiff), texte], check=True, timeout=20)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff), str(chemin)], check=True, timeout=20)
        except Exception as exc:
            logger.warning("Voix macOS indisponible (%s), robot muet.", exc)
            return False
    return True


def synthetiser(texte: str) -> Path | None:
    """Le WAV de `texte` : depuis le cache, sinon synthétisé (une clé de cache par moteur et par phrase)."""
    nom_moteur = moteur()
    chemin = DOSSIER_VOIX / f"{hashlib.sha256(f'{nom_moteur}|{texte}'.encode()).hexdigest()[:24]}.wav"
    if chemin.exists():
        return chemin
    DOSSIER_VOIX.mkdir(parents=True, exist_ok=True)
    provisoire = chemin.with_suffix(".tmp.wav")
    reussi = (nom_moteur.startswith("elevenlabs") and _elevenlabs(texte, nom_moteur.split(":", 1)[1], provisoire)) or _macos(
        texte, provisoire
    )
    if not reussi:
        return None
    provisoire.replace(chemin)  # renommage : une interruption ne laisse jamais un WAV tronqué en cache
    return chemin


def preparer(phrases: dict[str, str]) -> dict[str, Path | None]:
    """Toutes les répliques, synthétisées avant la démo."""
    return {cle: synthetiser(texte) for cle, texte in phrases.items()}


def durees(phrases: dict[str, str]) -> dict[str, float | None]:
    """Combien de temps le robot met à dire chaque réplique, en secondes (avec la voix configurée)."""
    return {cle: sf.info(str(chemin)).duration if chemin else None for cle, chemin in preparer(phrases).items()}
