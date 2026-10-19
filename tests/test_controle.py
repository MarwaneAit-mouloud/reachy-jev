"""Le contrôle de commande, contre un faux serveur OpenRouter (jamais de vrai réseau)."""

import json

import httpx
import pytest
from PIL import Image

from reachy_jev import openrouter
from reachy_jev.controle import QUESTIONS, controler, preparer_image, trancher
from reachy_jev.openrouter import creer_client
from reachy_jev.robot import reagir

IMAGE = "data:image/jpeg;base64,AAAA"


@pytest.fixture(autouse=True)
def sans_pause(monkeypatch):
    monkeypatch.setattr(openrouter, "_attendre", lambda secondes: None)


def client_qui_repond(reponse: dict, statut: int = 200, requetes: list | None = None) -> httpx.Client:
    def repondre(requete: httpx.Request) -> httpx.Response:
        if requetes is not None:
            requetes.append(json.loads(requete.content))
        return httpx.Response(statut, json=reponse)

    return creer_client("cle-de-test", transport=httpx.MockTransport(repondre))


def reponse_decision(p_bon: float, etat: str, confiance: float = 0.9) -> dict:
    return {
        "model": "typesafe/jev-1.14-20261014",
        "answers": {
            "bon_produit": {"type": "noul", "noul": p_bon},
            "etat": {"type": "choice", "choice": etat, "confidence": confiance, "probabilities": {}},
        },
        "usage": {"input_tokens": 2276, "output_tokens": 37, "cost": 0.0000956},
    }


@pytest.mark.parametrize(
    ("p_bon", "etat", "attendu"),
    [
        (0.95, "intact", "ok"),
        (0.95, "abime", "abime"),
        (0.1, "intact", "mauvais_produit"),
        (0.1, "abime", "mauvais_produit"),
    ],
)
def test_le_code_tranche(p_bon, etat, attendu):
    assert trancher(p_bon, etat) == attendu


def test_modele_de_decision_texte_et_image_dans_une_seule_requete():
    requetes: list = []
    controle = controler(
        client_qui_repond(reponse_decision(0.97, "intact"), requetes=requetes), "Jev 1.14", "1 canette de Coca-Cola", IMAGE
    )
    assert controle.verdict == "ok" and not controle.doute
    assert controle.cout_usd == pytest.approx(0.0000956) and controle.tokens_entree == 2276
    corps = requetes[0]
    assert corps["model"] == "typesafe/jev-1.14" and corps["questions"] == QUESTIONS
    assert corps["state"][0] == "Ligne de commande : 1 canette de Coca-Cola"
    assert corps["state"][-1] == {"type": "image_url", "image_url": {"url": IMAGE}}


def test_variante_texte_sans_image():
    requetes: list = []
    client = client_qui_repond(reponse_decision(0.02, "abime"), requetes=requetes)
    controle = controler(client, "Jev 1.14", "1 canette de Pepsi", description="Un Coca, complètement aplati.")
    assert controle.verdict == "mauvais_produit"
    corps = requetes[0]
    assert corps["state"] == ["Ligne de commande : 1 canette de Pepsi", "Ce que dit le préparateur : « Un Coca, complètement aplati. »"]
    assert "picker describes" in corps["questions"]["bon_produit"]["instructions"]
    with pytest.raises(ValueError):
        controler(client, "Jev 1.14", "x")


def test_doute_si_probabilite_au_milieu():
    controle = controler(client_qui_repond(reponse_decision(0.55, "intact")), "GPT-6 Luna Decisions", "x", IMAGE)
    assert controle.verdict == "ok" and controle.doute


def test_llm_generatif_json_contraint():
    reponse = {
        "choices": [{"message": {"content": '{"bon_produit": "non", "etat": "intact"}'}}],
        "usage": {"prompt_tokens": 1033, "completion_tokens": 12, "cost": 0.00015},
    }
    requetes: list = []
    controle = controler(client_qui_repond(reponse, requetes=requetes), "GPT-6 Luna (génératif)", "1 canette de Sprite", IMAGE)
    assert controle.verdict == "mauvais_produit" and controle.p_bon_produit == 0.0 and controle.tokens_entree == 1033
    corps = requetes[0]
    assert corps["response_format"]["json_schema"]["schema"]["properties"]["etat"]["enum"] == ["intact", "abime"]
    assert corps["messages"][1]["content"][1]["image_url"]["url"] == IMAGE


def test_une_erreur_reseau_devient_un_controle_en_erreur():
    controle = controler(client_qui_repond({"error": {"message": "nope"}}, statut=401), "Jev 1.14", "x", IMAGE)
    assert controle.verdict is None and "401" in controle.erreur


def test_reponse_inattendue_de_l_api_alpha():
    controle = controler(client_qui_repond({"answers": {}, "usage": {}}), "Jev 1.14", "x", IMAGE)
    assert controle.erreur and "inattendue" in controle.erreur


def test_preparer_image_respecte_les_limites_de_jev(tmp_path):
    chemin = tmp_path / "grande.jpg"
    Image.new("RGB", (3000, 4000), "red").save(chemin)
    url = preparer_image(chemin)
    assert url.startswith("data:image/jpeg;base64,")
    with pytest.raises(ValueError):
        preparer_image(Image.new("RGB", (200, 200)))


class RobotDeTest:
    """Remplace Reachy dans les tests : retient les gestes demandés, sans matériel."""

    def __init__(self):
        self.gestes = []

    def mouvement(self, nom):
        self.gestes.append(nom)

    def parler(self, chemin_wav):
        pass


def test_le_robot_reagit_au_verdict_ou_au_doute():
    robot = RobotDeTest()
    assert reagir(robot, controler(client_qui_repond(reponse_decision(0.1, "intact")), "Jev 1.14", "x", IMAGE)) == "mauvais_produit"
    assert reagir(robot, controler(client_qui_repond(reponse_decision(0.5, "intact")), "Jev 1.14", "x", IMAGE)) == "doute"
    assert robot.gestes == ["no1", "inquiring1"]  # il secoue la tête, puis la penche
