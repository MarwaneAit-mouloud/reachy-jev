"""Appels à OpenRouter : modèles de décision (API Decisions) et LLM (chat), avec latence et coût mesurés.

Aucune exception réseau ne sort d'ici : un échec est un `Appel` avec `erreur`, et le message dit quoi faire.
Le client est persistant (connexions réutilisées) : on ne mesure pas la poignée de main TLS à chaque appel.
"""

from __future__ import annotations

import random
import time
from typing import Any

import httpx
from pydantic import BaseModel

from reachy_jev.reglages import MSG_CLE_MANQUANTE, URL_CHAT, URL_DECISIONS, cle_openrouter

STATUTS_A_REESSAYER = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
OPTIONS_FACULTATIVES = ("reasoning", "temperature", "provider")  # retirées une à une si un modèle les refuse


class Appel(BaseModel):
    """Résultat d'un appel, succès ou échec."""

    donnees: dict[str, Any] | None = None
    latence_ms: float = 0.0  # durée de la dernière tentative : c'est celle qu'on compare
    cout_usd: float | None = None  # `usage.cost` renvoyé par OpenRouter, en dollars
    statut: int | None = None
    erreur: str | None = None


def creer_client(cle: str | None = None, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Client HTTP authentifié. Lève RuntimeError (avec la marche à suivre) s'il n'y a pas de clé."""
    cle = cle or cle_openrouter()
    if not cle:
        raise RuntimeError(MSG_CLE_MANQUANTE)
    return httpx.Client(
        headers={"Authorization": f"Bearer {cle}", "X-Title": "reachy-jev"},
        timeout=httpx.Timeout(60.0, connect=10.0),
        transport=transport,
    )


def _message_erreur(statut: int, corps: Any) -> str:
    """Traduit une erreur HTTP en consigne actionnable."""
    detail = ""
    if isinstance(corps, dict):
        err = corps.get("error")
        detail = str(err.get("message", err) if isinstance(err, dict) else err or "")
    if statut == 401:
        return "Clé OpenRouter refusée (401) : vérifie OPENROUTER_API_KEY dans .env."
    if statut == 402:
        return "Crédit OpenRouter insuffisant (402) : recharge sur https://openrouter.ai/credits."
    if statut == 404:
        return f"Modèle ou endpoint introuvable (404) : {detail or 'vérifie le nom du modèle dans reglages.py'}."
    if statut == 429:
        return "Limite de débit atteinte (429), même après plusieurs essais : attends un peu."
    return f"Erreur HTTP {statut} : {detail[:300] or 'sans détail'}."


def _attendre(secondes: float) -> None:
    """Pause entre deux essais (isolée pour que les tests l'annulent sans toucher à time.sleep)."""
    time.sleep(secondes)


def poster(client: httpx.Client, url: str, corps: dict[str, Any], tentatives: int = 3) -> Appel:
    """POST JSON, avec reprises sur les erreurs passagères. Ne lève jamais."""
    dernier = Appel(erreur="Aucune tentative effectuée.")
    for essai in range(1, tentatives + 1):
        debut = time.perf_counter()
        try:
            reponse = client.post(url, json=corps)
        except httpx.TimeoutException:
            dernier = Appel(latence_ms=(time.perf_counter() - debut) * 1000, erreur="Délai dépassé : le serveur n'a pas répondu à temps.")
        except httpx.HTTPError as exc:
            dernier = Appel(
                latence_ms=(time.perf_counter() - debut) * 1000,
                erreur=f"Réseau indisponible ({type(exc).__name__}) : vérifie ta connexion.",
            )
        else:
            latence_ms = (time.perf_counter() - debut) * 1000
            try:
                donnees = reponse.json()
            except ValueError:
                donnees = None
            statut = reponse.status_code
            # OpenRouter peut renvoyer un 200 dont le corps est une erreur du fournisseur.
            erreur_corps = donnees.get("error") if isinstance(donnees, dict) else None
            if erreur_corps and statut == 200:
                code = erreur_corps.get("code") if isinstance(erreur_corps, dict) else None
                statut = code if isinstance(code, int) else 502
            if statut == 200 and isinstance(donnees, dict):
                cout = (donnees.get("usage") or {}).get("cost")
                return Appel(
                    donnees=donnees, latence_ms=latence_ms, statut=200, cout_usd=float(cout) if isinstance(cout, (int, float)) else None
                )
            dernier = Appel(
                statut=statut,
                latence_ms=latence_ms,
                donnees=donnees if isinstance(donnees, dict) else None,
                erreur=_message_erreur(statut, donnees),
            )
            if statut not in STATUTS_A_REESSAYER:
                break
        if essai < tentatives:
            _attendre(0.6 * 2 ** (essai - 1) + random.uniform(0, 0.2))
    return dernier


def decider(client: httpx.Client, modele: str, state: Any, questions: dict[str, Any]) -> Appel:
    """Une requête à un modèle de décision (Jev, GPT-6 Luna Decisions…) : `donnees["answers"][nom_de_question]`."""
    return poster(client, URL_DECISIONS, {"model": modele, "state": state, "questions": questions})


def chat(
    client: httpx.Client,
    config: dict[str, Any],
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None = None,
    max_tokens: int = 300,
) -> Appel:
    """Une requête à un LLM. `config` = {"model": ..., options}. Avec `schema`, la sortie est contrainte en JSON.

    Si le modèle refuse une option (400/404/422), on la retire et on retente, plutôt que d'échouer.
    """
    corps: dict[str, Any] = {**config, "messages": messages, "max_tokens": max_tokens, "usage": {"include": True}}
    if schema is not None:
        corps["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": schema}}
        corps["provider"] = {"require_parameters": True}  # seulement les fournisseurs qui respectent le schéma
    a_retirer = [option for option in OPTIONS_FACULTATIVES if option in corps]
    while True:
        appel = poster(client, URL_CHAT, corps)
        if not (appel.erreur and appel.statut in (400, 404, 422) and a_retirer):
            return appel
        corps.pop(a_retirer.pop(0))


def texte(appel: Appel) -> str | None:
    """Le texte renvoyé par un LLM, ou None."""
    try:
        return appel.donnees["choices"][0]["message"]["content"] or None  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return None
