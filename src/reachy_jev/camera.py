"""Mesures sur le vrai robot, caméra comprise. Aucune image n'est enregistrée : seules les durées le sont.

- `mesurer` : la perception seule (image neuve de la caméra, puis encodage pour l'IA) ;
- `mesurer_bout_en_bout` : le pipeline complet pour chaque modèle (capture → encodage → inférence → réponse vocale), sur
  de vraies prises de vue. L'encodage est le même pour tous les modèles ; ce qui diffère est côté serveur (découpage de
  l'image en tokens), et c'est compris dans l'inférence : on relève donc aussi les tokens d'entrée de chaque modèle.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from reachy_jev import voix
from reachy_jev.controle import controler, preparer_image
from reachy_jev.reglages import CERVEAUX, DOSSIER_RESULTATS
from reachy_jev.robot import PHRASES, Reachy, reaction_pour

FICHIER = DOSSIER_RESULTATS / "capture.json"
FICHIER_BOUT_EN_BOUT = DOSSIER_RESULTATS / "bout_en_bout.json"


def mesurer(robot: Reachy, images: int = 20) -> dict[str, Any]:
    """Première image après connexion, puis `images` prises de vue neuves, chacune préparée pour l'IA."""
    debut = time.perf_counter()
    precedente = robot.photo(attente_max_s=10.0)
    premiere_ms = (time.perf_counter() - debut) * 1000
    captures, preparations = [], []
    for _ in range(images):
        debut = time.perf_counter()
        image = robot.photo(differente_de=precedente)
        milieu = time.perf_counter()
        preparer_image(image)
        captures.append((milieu - debut) * 1000)
        preparations.append((time.perf_counter() - milieu) * 1000)
        precedente = image
    return {
        "date": date.today().isoformat(),
        "images": images,
        "resolution": list(precedente.size),
        "premiere_image_ms": round(premiere_ms, 1),
        "capture_mediane_ms": round(statistics.median(captures), 1),
        "capture_max_ms": round(max(captures), 1),
        "preparation_mediane_ms": round(statistics.median(preparations), 1),
    }


def _mediane(valeurs: list[float]) -> float | None:
    return round(statistics.median(valeurs), 1) if valeurs else None


def mesurer_bout_en_bout(
    robot: Reachy,
    client: httpx.Client,
    commande: str,
    cycles: int = 8,
    cerveaux: list[str] | None = None,
) -> dict[str, Any]:
    """Pour chaque modèle et chaque cycle : une prise de vue neuve, son encodage, l'inférence, puis la durée de la phrase
    que dirait le robot. L'ordre des modèles tourne d'un cycle à l'autre ; appels un par un, comme sur l'expérimentation."""
    cerveaux = cerveaux or list(CERVEAUX)
    parole_s = voix.durees(PHRASES)
    precedente = robot.photo(attente_max_s=10.0)
    for nom in cerveaux:  # chauffe : connexions ouvertes, rien de compté
        controler(client, nom, commande, image_url=preparer_image(precedente))
    mesures: dict[str, list[dict[str, Any]]] = {nom: [] for nom in cerveaux}
    for cycle in range(cycles):
        decalage = cycle % len(cerveaux)
        for nom in cerveaux[decalage:] + cerveaux[:decalage]:
            debut = time.perf_counter()
            image = robot.photo(differente_de=precedente)
            capture = time.perf_counter()
            image_url = preparer_image(image)
            encodage = time.perf_counter()
            resultat = controler(client, nom, commande, image_url=image_url)
            mesures[nom].append(
                {
                    "capture_ms": (capture - debut) * 1000,
                    "encodage_ms": (encodage - capture) * 1000,
                    "image_ko": len(image_url) * 3 / 4 / 1024,
                    "inference_ms": resultat.latence_ms,
                    "reaction": reaction_pour(resultat),
                    "parole_s": parole_s.get(reaction_pour(resultat)),
                    "cout_usd": resultat.cout_usd,
                    "tokens_entree": resultat.tokens_entree,
                    "erreur": resultat.erreur,
                }
            )
            precedente = image
    modeles = {}
    for nom, lignes in mesures.items():
        valides = [ligne for ligne in lignes if not ligne["erreur"]]

        def colonne(cle: str, lignes: list[dict[str, Any]] = valides) -> list[float]:
            return [ligne[cle] for ligne in lignes if ligne[cle] is not None]

        etapes = {cle: _mediane(colonne(cle)) for cle in ("capture_ms", "encodage_ms", "inference_ms")}
        parole = _mediane(colonne("parole_s"))
        modeles[nom] = {
            "cycles": len(lignes),
            "erreurs": len(lignes) - len(valides),
            "reactions": dict(Counter(ligne["reaction"] for ligne in lignes).most_common()),
            "capture_mediane_ms": etapes["capture_ms"],
            "encodage_mediane_ms": etapes["encodage_ms"],
            "inference_mediane_ms": etapes["inference_ms"],
            "inference_max_ms": round(max(colonne("inference_ms")), 1) if valides else None,
            "reponse_vocale_mediane_s": parole,
            "total_median_s": round(sum(v or 0 for v in etapes.values()) / 1000 + (parole or 0), 2),
            "image_ko": _mediane(colonne("image_ko")),
            "tokens_entree_median": _mediane(colonne("tokens_entree")),
            "cout_moyen_usd": round(statistics.fmean(colonne("cout_usd")), 8) if colonne("cout_usd") else None,
            "inference_ms_par_cycle": [round(ligne["inference_ms"]) for ligne in lignes],
            "erreurs_detail": sorted({ligne["erreur"][:160] for ligne in lignes if ligne["erreur"]}),
        }
    return {
        "date": date.today().isoformat(),
        "commande": commande,
        "resolution": list(precedente.size),
        "modeles": modeles,
    }


def ecrire(mesure: dict[str, Any], chemin: Path = FICHIER) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(mesure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return chemin


def lire(chemin: Path = FICHIER) -> dict[str, Any] | None:
    """La dernière mesure faite sur le robot, ou None si elle n'existe pas encore."""
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else None
