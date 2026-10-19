"""La reconnaissance vocale du contrôle à la voix, sans réseau : requête envoyée, texte rendu, comparaison au texte prononcé."""

import base64
import json

import httpx

from reachy_jev.openrouter import creer_client
from reachy_jev.reglages import MODELE_STT
from reachy_jev.stt import similarite, transcrire


def test_similarite_compare_les_mots_pas_la_typographie():
    assert similarite("Un Coca-Cola, la canette est toute froissée.", "un coca cola la canette est toute froissee") == 1.0
    assert similarite("la canette est toute froissée", "la canette est toute propre") < 0.95  # le sens s'inverse
    assert similarite("une canette", None) == 0.0


def test_transcrire_envoie_l_audio_et_rend_le_texte(tmp_path):
    wav = tmp_path / "phrase.wav"
    wav.write_bytes(b"RIFF-faux-wav")
    requetes = []

    def repondre(requete: httpx.Request) -> httpx.Response:
        requetes.append(json.loads(requete.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": " Une canette de Pepsi. "}}], "usage": {"cost": 0.00001}})

    texte, appel = transcrire(creer_client("cle-de-test", transport=httpx.MockTransport(repondre)), wav)
    assert texte == "Une canette de Pepsi." and appel.erreur is None
    corps = requetes[0]
    audio = corps["messages"][0]["content"][1]
    assert corps["model"] == MODELE_STT and audio["type"] == "input_audio" and audio["input_audio"]["format"] == "wav"
    assert base64.b64decode(audio["input_audio"]["data"]) == b"RIFF-faux-wav"
