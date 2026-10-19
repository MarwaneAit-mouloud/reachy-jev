"""Le cas d'usage : Reachy contrôle une commande, avec une photo ET un texte.

Un préparateur montre un article à la caméra. Reachy connaît la ligne de commande (texte) et voit l'article (photo) :
- « ok » : c'est le bon produit, intact → il hoche la tête ;
- « mauvais_produit » : ce n'est pas l'article commandé → il secoue la tête ;
- « abime » : c'est le bon produit, mais abîmé → il fait la moue.
La même photo peut être bonne pour une commande et mauvaise pour une autre : il faut le texte et l'image.

Les cerveaux ne rendent pas le verdict eux-mêmes. On leur pose deux questions simples (est-ce le bon produit ? dans
quel état ?) et le code tranche : « le modèle mesure, le code décide ». Les deux modèles de décision reçoivent
exactement la même requête ; le LLM génératif reçoit les mêmes consignes et répond dans un JSON imposé.
Consignes en anglais (la langue où Jev est le plus fiable), ligne de commande en français.

Variante texte : la photo est remplacée par ce que dit le préparateur (« C'est un Coca, complètement aplati »).
Elle sert à vérifier les promesses de coût là où TypeSafe les fait, sur du texte.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps
from pydantic import BaseModel

from reachy_jev import openrouter
from reachy_jev.reglages import CERVEAUX, COTE_MAX_IMAGE, DOUTE_BAS, DOUTE_HAUT, SEUIL_CONFIANCE_ETAT

VERDICTS = {"ok": "bon produit, intact", "mauvais_produit": "mauvais produit", "abime": "bon produit, mais abîmé"}
ETATS = {
    "intact": "Undamaged: the can or bottle keeps its normal shape.",
    "abime": "Damaged: dented, crushed, deformed or torn.",
}


def questions(avec_image: bool = True) -> dict[str, Any]:
    """Les deux questions, sur la photo ou sur ce que décrit le préparateur."""
    article = "the product in the photo" if avec_image else "the item the picker describes"
    return {
        "bon_produit": {
            "type": "noul",
            "instructions": f"Is {article} the product named in the order line (same brand and same drink)?",
        },
        "etat": {"type": "choice", "instructions": f"What condition is {article} in?", "criteria": ETATS},
    }


QUESTIONS = questions(avec_image=True)
PROMPT_LLM = (
    "You check orders in a warehouse. You receive an order line and a photo of the item shown to the camera. "
    "Answer each question by applying the criteria literally. Reply with a single JSON object matching the schema."
)


class Controle(BaseModel):
    """Ce qu'un cerveau a vu, le verdict qu'en tire le code, et ce que ça a coûté."""

    cerveau: str
    verdict: str | None = None  # une clé de VERDICTS
    p_bon_produit: float | None = None  # probabilité du « oui » (1 ou 0 pour le LLM, qui répond oui ou non)
    etat: str | None = None
    confiance_etat: float | None = None
    doute: bool = False  # le robot demande de remontrer l'article au lieu de trancher
    latence_ms: float = 0.0
    cout_usd: float | None = None
    tokens_entree: int | None = None
    erreur: str | None = None


def preparer_image(source: Path | Image.Image) -> str:
    """Photo → URL `data:` JPEG en base64, redressée et ramenée à COTE_MAX_IMAGE px : la même pour tous les cerveaux."""
    image = Image.open(source) if isinstance(source, Path) else source
    image = ImageOps.exif_transpose(image).convert("RGB")  # l'API n'applique pas l'orientation EXIF
    image.thumbnail((COTE_MAX_IMAGE, COTE_MAX_IMAGE))
    if min(image.size) < 256:
        raise ValueError(f"Photo trop petite ({image.size[0]}×{image.size[1]}) : Jev demande au moins 256 px de côté.")
    tampon = io.BytesIO()
    image.save(tampon, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(tampon.getvalue()).decode()


def trancher(p_bon_produit: float, etat: str) -> str:
    """Le verdict, décidé par le code à partir des deux réponses."""
    if p_bon_produit < 0.5:
        return "mauvais_produit"
    return "abime" if etat == "abime" else "ok"


def etat_decision(commande: str, image_url: str | None = None, description: str | None = None) -> list[Any]:
    """Le `state` envoyé aux modèles de décision : la ligne de commande, puis l'image (au premier niveau, seul endroit
    lu par tous) ou ce que dit le préparateur."""
    if image_url is not None:
        return [
            f"Ligne de commande : {commande}",
            "Photo de l'article montré à la caméra :",
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    return [f"Ligne de commande : {commande}", f"Ce que dit le préparateur : « {description} »"]


def _tokens(donnees: dict[str, Any]) -> int | None:
    usage = donnees.get("usage") or {}
    valeur = usage.get("input_tokens", usage.get("prompt_tokens"))
    return int(valeur) if isinstance(valeur, (int, float)) else None


def _par_decision(client: httpx.Client, nom: str, modele: str, commande: str, image_url: str | None, description: str | None) -> Controle:
    appel = openrouter.decider(client, modele, etat_decision(commande, image_url, description), questions(image_url is not None))
    if appel.erreur:
        return Controle(cerveau=nom, erreur=appel.erreur, latence_ms=appel.latence_ms)
    donnees = appel.donnees or {}
    communs = dict(cerveau=nom, latence_ms=appel.latence_ms, cout_usd=appel.cout_usd, tokens_entree=_tokens(donnees))
    reponses = donnees.get("answers") or {}
    p = (reponses.get("bon_produit") or {}).get("noul")
    etat = (reponses.get("etat") or {}).get("choice")
    confiance = (reponses.get("etat") or {}).get("confidence")
    if not isinstance(p, (int, float)) or etat not in ETATS:
        return Controle(erreur=f"réponse inattendue : {str(reponses)[:150]} (l'API est en alpha)", **communs)
    doute = DOUTE_BAS < p < DOUTE_HAUT or (isinstance(confiance, (int, float)) and confiance < SEUIL_CONFIANCE_ETAT)
    return Controle(verdict=trancher(p, etat), p_bon_produit=p, etat=etat, confiance_etat=confiance, doute=doute, **communs)


def schema_llm() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"bon_produit": {"type": "string", "enum": ["oui", "non"]}, "etat": {"type": "string", "enum": list(ETATS)}},
        "required": ["bon_produit", "etat"],
        "additionalProperties": False,
    }


def messages_llm(commande: str, image_url: str | None = None, description: str | None = None) -> list[dict[str, Any]]:
    """Exactement les consignes et critères envoyés aux modèles de décision, mis en texte, plus la même image (ou description)."""
    q = questions(image_url is not None)
    criteres = "\n".join(f"  - {etat}: {texte_etat}" for etat, texte_etat in ETATS.items())
    entree = "" if image_url is not None else f"Ce que dit le préparateur : « {description} »\n"
    texte = (
        f"Ligne de commande : {commande}\n{entree}\n"
        f'bon_produit: {q["bon_produit"]["instructions"]} Answer "oui" or "non".\n\n'
        f"etat: {q['etat']['instructions']}\n{criteres}"
    )
    contenu: Any = texte if image_url is None else [{"type": "text", "text": texte}, {"type": "image_url", "image_url": {"url": image_url}}]
    return [{"role": "system", "content": PROMPT_LLM}, {"role": "user", "content": contenu}]


def _par_llm(
    client: httpx.Client, nom: str, config: dict[str, Any], commande: str, image_url: str | None, description: str | None
) -> Controle:
    appel = openrouter.chat(client, config, messages_llm(commande, image_url, description), schema=schema_llm(), max_tokens=400)
    if appel.erreur:
        return Controle(cerveau=nom, erreur=appel.erreur, latence_ms=appel.latence_ms)
    donnees = appel.donnees or {}
    communs = dict(cerveau=nom, latence_ms=appel.latence_ms, cout_usd=appel.cout_usd, tokens_entree=_tokens(donnees))
    contenu = openrouter.texte(appel) or ""
    try:
        objet = json.loads(contenu[contenu.find("{") : contenu.rfind("}") + 1])
    except json.JSONDecodeError:
        objet = None
    if not isinstance(objet, dict) or objet.get("bon_produit") not in ("oui", "non") or objet.get("etat") not in ETATS:
        return Controle(erreur=f"JSON illisible ou hors options : {contenu[:120]!r}", **communs)
    p = 1.0 if objet["bon_produit"] == "oui" else 0.0
    return Controle(verdict=trancher(p, objet["etat"]), p_bon_produit=p, etat=objet["etat"], **communs)


def controler(client: httpx.Client, nom: str, commande: str, image_url: str | None = None, description: str | None = None) -> Controle:
    """Un contrôle complet par le cerveau `nom` (une clé de CERVEAUX), sur une photo OU sur ce que dit le préparateur.
    Ne lève jamais : un échec est un Controle avec `erreur`."""
    if (image_url is None) == (description is None):
        raise ValueError("Donne soit une image, soit une description.")
    config = CERVEAUX[nom]
    if config["type"] == "decision":
        return _par_decision(client, nom, config["model"], commande, image_url, description)
    return _par_llm(client, nom, {k: v for k, v in config.items() if k != "type"}, commande, image_url, description)
