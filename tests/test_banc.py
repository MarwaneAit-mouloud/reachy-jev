"""Les chiffres de l'expérimentation et les mesures sur le robot, sur des données inventées (aucun appel réseau, aucun robot)."""

import itertools
import json

import httpx
from PIL import Image

from reachy_jev import camera
from reachy_jev.banc import (
    _fois,
    charger_lignes,
    ecrire_resultats,
    note_transcription,
    perception_mesuree,
    remplacer_par_robot,
    resumer,
    sources,
    tableau_markdown,
)
from reachy_jev.openrouter import creer_client


def ligne(cerveau, verdict, attendu, latence, mode="photo", cout=0.0001, p=0.9, etat="intact", erreur=None, entree="a.jpg"):
    return {
        "date": "2026-10-17",
        "mode": mode,
        "photo": "a.jpg",
        "commande": "x",
        "entree": entree,
        "attendu": attendu,
        "etat_photo": "intact",
        "cerveau": cerveau,
        "verdict": verdict,
        "juste": verdict == attendu,
        "p_bon_produit": p,
        "etat": etat,
        "confiance_etat": 0.9,
        "doute": False,
        "preparation_ms": 40.0 if mode == "photo" else 0.0,
        "latence_ms": latence,
        "parole_s": 2.0,
        "cout_usd": cout,
        "tokens_entree": 1000,
        "erreur": erreur,
    }


LIGNES = [
    ligne("A", "ok", "ok", 300),
    ligne("A", "ok", "ok", 500),
    ligne("A", "mauvais_produit", "ok", 400, p=0.2),
    ligne("B", None, "ok", 0, cout=None, p=None, etat=None, erreur="réseau"),
    ligne("B", "ok", "ok", 3000),
    ligne("A", "ok", "abime", 200, mode="voix", cout=0.00001, entree="la canette est toute propre"),
    ligne("B", "ok", "ok", 1500, mode="voix", entree="une canette de Pepsi"),
]
CAPTURE = {
    "date": "2026-10-19",
    "images": 20,
    "resolution": [1280, 720],
    "premiere_image_ms": 900.0,
    "capture_mediane_ms": 35.0,
    "capture_max_ms": 60.0,
    "preparation_mediane_ms": 12.0,
}
STT = {
    "modele": "google/gemini-2.5-flash-lite",
    "latence_mediane_ms": 950,
    "phrases": [
        {"texte": "la canette est toute froissée", "transcription": "la canette est toute propre", "similarite": 0.82},
        {"texte": "une canette de Pepsi", "transcription": "une canette de Pepsi", "similarite": 1.0},
    ],
}
BOUT_EN_BOUT = {
    "modeles": {
        "A": {
            "cycles": 8,
            "capture_mediane_ms": 20.0,
            "encodage_mediane_ms": 10.0,
            "inference_mediane_ms": 350.0,
            "inference_max_ms": 600.0,
            "reponse_vocale_mediane_s": 2.0,
            "cout_moyen_usd": 0.0002,
            "tokens_entree_median": 1980.0,
        }
    }
}


def test_resume_par_mode_et_par_modele():
    a, b, a_voix, _ = resumer(LIGNES, ["A", "B"])
    assert (a["mode"], a["justes"], a["controles"], a["latence_mediane_ms"], a["latence_max_ms"]) == ("photo", 2, 3, 400, 500)
    assert a["produit_juste"] == 2 and a["cout_million_usd"] == 100
    assert (b["erreurs"], b["justes"], b["latence_mediane_ms"]) == (1, 1, 3000)  # l'erreur compte comme fausse, pas dans la latence
    assert a["etapes_s"] == {"perception": 0.04, "inference": 0.4, "action": 2.0} and abs(a["total_s"] - 2.44) < 1e-9
    assert (a_voix["mode"], a_voix["etapes_s"]["perception"], round(a_voix["cout_million_usd"])) == ("voix", 0.0, 10)


def test_perception_mesuree_a_part_pour_chaque_mode():
    perception = perception_mesuree(CAPTURE, STT)
    assert abs(perception["photo"] - 0.047) < 1e-9 and perception["voix"] == 0.95
    a, _, a_voix, _ = resumer(LIGNES, ["A", "B"], perception=perception)
    assert abs(a["etapes_s"]["perception"] - 0.047) < 1e-9 and a_voix["etapes_s"]["perception"] == 0.95


def test_le_robot_remplace_les_latences_de_la_vision_seulement():
    a, b, a_voix, _ = remplacer_par_robot(resumer(LIGNES, ["A", "B"]), BOUT_EN_BOUT)
    assert a["source"] == "robot" and a["etapes_s"] == {"perception": 0.03, "inference": 0.35, "action": 2.0}
    assert (
        a["justes"] == 2 and round(a["cout_million_usd"]) == 200 and a["tokens_moyens"] == 1980.0
    )  # l'exactitude reste celle des contrôles
    assert b["source"] == a_voix["source"] == "experimentation"


def test_erreurs_a_la_voix_rapportees_a_la_transcription():
    assert note_transcription(LIGNES, STT) == "toutes les erreurs (1) portent sur une phrase mal transcrite par la STT"
    assert note_transcription(LIGNES, None) is None


def test_sources_sans_date():
    pieds = sources(STT, BOUT_EN_BOUT)
    assert all(pied.startswith("Source : Expérimentations avec OpenRouter") and "2026" not in pied for pied in pieds.values())
    assert "gemini-2.5-flash-lite" in pieds["voix"] and "médiane de 8 cycles" in pieds["photo"]


def test_fichiers_ecrits_et_relus(tmp_path):
    resume = resumer(LIGNES, ["A", "B"], perception=perception_mesuree(CAPTURE, STT))
    assert "**2/3**" in tableau_markdown(resume, "photo") and "**0/1**" in tableau_markdown(resume, "voix")
    chemins = ecrire_resultats(LIGNES, resume, tmp_path, notes={"voix": note_transcription(LIGNES, STT)})
    assert [c.name for c in chemins] == ["banc.csv", "resume.md", "latence.png", "linkedin.png", "linkedin_voix.png"]
    assert all(c.exists() and c.stat().st_size > 0 for c in chemins)
    assert "mal transcrite" in (tmp_path / "resume.md").read_text(encoding="utf-8")
    relues = charger_lignes(tmp_path / "banc.csv")
    assert relues[0]["latence_ms"] == 300.0 and relues[0]["juste"] is True and relues[3]["erreur"] == "réseau"
    assert relues[5]["entree"] == "la canette est toute propre"


def test_rapport_jamais_enjolive():
    assert _fois(7.77) == "près de 8 fois" and _fois(7.2) == "plus de 7 fois"


class FauxRobot:
    """Une caméra qui renvoie une image différente à chaque appel."""

    def __init__(self):
        self.couleurs = itertools.cycle([(10, 10, 10), (200, 200, 200)])

    def photo(self, attente_max_s=3.0, differente_de=None):
        return Image.new("RGB", (640, 480), next(self.couleurs))


def test_mesure_de_capture_sans_robot(tmp_path):
    mesure = camera.mesurer(FauxRobot(), images=3)
    assert mesure["images"] == 3 and mesure["resolution"] == [640, 480]
    assert mesure["capture_mediane_ms"] >= 0 and mesure["preparation_mediane_ms"] > 0
    chemin = camera.ecrire(mesure, tmp_path / "capture.json")
    assert camera.lire(chemin) == mesure and camera.lire(tmp_path / "absent.json") is None


def test_bout_en_bout_sans_robot(monkeypatch):
    monkeypatch.setattr(camera.voix, "durees", lambda phrases: dict.fromkeys(phrases, 2.0))
    requetes = []

    def repondre(requete: httpx.Request) -> httpx.Response:
        requetes.append(json.loads(requete.content))
        reponses = {"bon_produit": {"type": "noul", "noul": 0.95}, "etat": {"type": "choice", "choice": "intact", "confidence": 0.9}}
        return httpx.Response(200, json={"answers": reponses, "usage": {"input_tokens": 1980, "cost": 0.0001}})

    client = creer_client("cle-de-test", transport=httpx.MockTransport(repondre))
    mesure = camera.mesurer_bout_en_bout(FauxRobot(), client, "1 canette de Coca-Cola", cycles=3, cerveaux=["Jev 1.14"])
    jev = mesure["modeles"]["Jev 1.14"]
    assert len(requetes) == 1 + 3  # une chauffe, puis un appel par cycle
    assert (jev["cycles"], jev["erreurs"], jev["tokens_entree_median"], jev["reponse_vocale_mediane_s"]) == (3, 0, 1980, 2.0)
    assert jev["reactions"] == {"ok": 3}
    assert jev["capture_mediane_ms"] >= 0 and jev["encodage_mediane_ms"] > 0 and len(jev["inference_ms_par_cycle"]) == 3
