"""Le Reachy Mini, piloté par le réseau : sa caméra, ses gestes et sa voix.

Points de connexion vérifiés sur un Reachy Mini Wireless :
- sonde du démon (GET /api/daemon/status) pour distinguer « robot éteint » de « SDK en vrac » ;
- réacquisition du flux média avant de se connecter ;
- enable_motors() PUIS wake_up() : sans ça, le démon accepte les ordres et rien ne bouge ;
- images de la caméra en BGR, qui mettent jusqu'à ~2 s à arriver après la connexion.
Les gestes sont des mouvements enregistrés de la bibliothèque de Pollen Robotics.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from math import gcd
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf
from PIL import Image
from scipy.signal import resample_poly

from reachy_jev.controle import Controle

logger = logging.getLogger(__name__)

PORT_DEMON = 8000
BIBLIOTHEQUE = "pollen-robotics/reachy-mini-emotions-library"
# Réaction (verdict, doute ou erreur) → mouvement enregistré de Pollen, et phrase dite en même temps.
MOUVEMENTS = {"ok": "yes1", "mauvais_produit": "no1", "abime": "displeased1", "doute": "inquiring1", "erreur": "confused1"}
PHRASES = {
    "ok": "C'est bon, article validé !",
    "mauvais_produit": "Attention, ce n'est pas le bon produit.",
    "abime": "Le produit est abîmé, on le met de côté.",
    "doute": "Je ne suis pas sûr. Tu peux me le remontrer ?",
    "erreur": "Je n'ai pas réussi à vérifier.",
}


class RobotIndisponible(RuntimeError):
    """Le robot ne répond pas ; le message dit quoi vérifier."""


class Reachy:
    """Un Reachy Mini joignable par le réseau."""

    def __init__(self, hote: str) -> None:
        self.hote = hote
        self._mini: Any = None
        self._mouvements: Any = None

    def ouvrir(self) -> str:
        """Connecte, active les moteurs et réveille le robot. Lève RobotIndisponible avec la marche à suivre."""
        try:
            etat = httpx.get(f"http://{self.hote}:{PORT_DEMON}/api/daemon/status", timeout=5).json()
        except Exception:
            raise RobotIndisponible(
                f"Aucune réponse de {self.hote}:{PORT_DEMON}. Le robot est-il allumé et sur le même WiFi que ce Mac ? "
                "Si le nom .local ne résout pas, mets son adresse IP dans REACHY_HOTE (fichier .env)."
            ) from None
        from reachy_mini import ReachyMini

        try:  # un client précédent tué net laisse le flux média « relâché » côté robot
            httpx.post(f"http://{self.hote}:{PORT_DEMON}/api/media/acquire", timeout=5)
        except Exception as exc:
            logger.warning("Réacquisition du média impossible (%s), on tente quand même.", exc)
        try:
            self._mini = ReachyMini(host=self.hote, connection_mode="network")
            self._mini.enable_motors()
            self._mini.wake_up()
        except Exception as exc:
            self._mini = None
            raise RobotIndisponible(
                f"Connexion impossible ({exc}). Le démon répond pourtant : "
                "une autre application (app Reachy, photobooth…) le tient peut-être."
            ) from exc
        return f"Connecté à {self.hote} (démon {etat.get('state', '?')}, version {etat.get('version', '?')})."

    def photo(self, attente_max_s: float = 3.0, differente_de: Image.Image | None = None) -> Image.Image:
        """Une image de la caméra du robot. Lève RobotIndisponible si rien n'arrive.

        `get_frame()` renvoie la dernière image reçue du flux, pas une nouvelle prise de vue. Avec `differente_de`,
        on attend une image neuve : c'est le vrai temps de perception qu'on mesure.
        """
        reference = _empreinte(differente_de) if differente_de is not None else None
        fin = time.monotonic() + attente_max_s
        while time.monotonic() < fin:
            trame_bgr = self._mini.media.get_frame()
            if trame_bgr is not None:
                image = Image.fromarray(np.asarray(trame_bgr)[:, :, ::-1])
                if reference is None or not np.array_equal(_empreinte(image), reference):
                    return image
            time.sleep(0.002)
        raise RobotIndisponible("La caméra du robot n'envoie aucune image : une autre application la tient peut-être.")

    def mouvement(self, nom: str) -> None:
        if self._mouvements is None:
            from reachy_mini.motion.recorded_move import RecordedMoves

            self._mouvements = RecordedMoves(BIBLIOTHEQUE)
        try:
            self._mini.play_move(self._mouvements.get(nom), initial_goto_duration=0.4)
        except Exception as exc:
            logger.warning("Mouvement « %s » impossible (%s), ignoré.", nom, exc)

    def parler(self, chemin_wav: Path) -> None:
        """Pousse le WAV au haut-parleur du robot en flux : `play_sound` coûte ~3 s de transfert par appel."""
        try:
            media = self._mini.media
            donnees, frequence = sf.read(str(chemin_wav), dtype="float32", always_2d=False)
            if donnees.ndim > 1:
                donnees = donnees.mean(axis=1)
            cible = 16000
            with contextlib.suppress(Exception):  # certains démons ne donnent pas la fréquence : 16 kHz par défaut
                cible = int(media.get_output_audio_samplerate()) or cible
            if frequence != cible:
                g = gcd(cible, frequence)
                donnees = resample_poly(donnees, cible // g, frequence // g).astype(np.float32)
            stereo = np.stack([donnees, donnees], axis=1)
            pas = cible // 4  # paquets de 0,25 s
            media.start_playing()
            try:
                for debut in range(0, len(stereo), pas):
                    paquet = stereo[debut : debut + pas]
                    media.push_audio_sample(paquet)
                    time.sleep(len(paquet) / cible)
            finally:
                media.stop_playing()
        except Exception as exc:
            logger.warning("Le robot n'a pas pu parler (%s), il reste muet.", exc)

    def dormir(self) -> None:
        """Position de sommeil puis moteurs coupés (moteurs sous couple = chaleur)."""
        self._mini.goto_sleep()
        self._mini.disable_motors()

    def fermer(self) -> None:
        if self._mini is not None:
            try:
                self._mini.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("Fermeture imparfaite (%s), on continue.", exc)
            self._mini = None


def _empreinte(image: Image.Image) -> np.ndarray:
    """Échantillon de pixels : suffit pour savoir si deux images du flux sont identiques, sans tout comparer."""
    return np.asarray(image)[::40, ::40]


def reaction_pour(controle: Controle) -> str:
    """« ok », « mauvais_produit », « abime », « doute » ou « erreur »."""
    return "erreur" if controle.erreur else "doute" if controle.doute else controle.verdict or "erreur"


def reagir(robot: Reachy, controle: Controle, voix: dict[str, Path | None] | None = None) -> str:
    """Le robot dit sa phrase et joue son geste en même temps. Renvoie le nom de la réaction."""
    reaction = reaction_pour(controle)
    wav = (voix or {}).get(reaction)
    parole = threading.Thread(target=robot.parler, args=(wav,)) if wav else None
    if parole:
        parole.start()
    robot.mouvement(MOUVEMENTS[reaction])
    if parole:
        parole.join()
    return reaction
