"""L'expérimentation : chaque contrôle de data/controle.json passé aux trois modèles, par la vision (photo de l'article)
puis par la voix (ce que la reconnaissance vocale a compris du préparateur), puis chiffres, tableaux et graphiques.

Mesures honnêtes : un appel de chauffe par modèle (non compté), appels un par un (jamais en parallèle), ordre des
modèles qui tourne d'un contrôle à l'autre, même entrée pour tous. Aucun chiffre n'est écrit à la main.
La latence est détaillée de bout en bout, en trois étapes :
- perception : capture caméra + encodage (vision), ou reconnaissance vocale STT (voix), mesurées à part ;
- inférence : l'appel au modèle de décision ou au LLM, réseau compris ;
- action : la réponse vocale du robot (TTS), qui part dès que l'inférence est terminée.
Si le pipeline complet a été mesuré sur le robot (`reachy-jev bout-en-bout`), il remplace ces valeurs pour la vision.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from reachy_jev import voix
from reachy_jev.controle import controler, preparer_image
from reachy_jev.reglages import CERVEAUX, DOSSIER_DONNEES, DOSSIER_RESULTATS
from reachy_jev.robot import PHRASES, reaction_pour
from reachy_jev.stt import SEUIL_EXACTE

MODES = {"photo": "Par la vision (photo de l'article)", "voix": "Par la voix (le préparateur décrit l'article)"}
PERCEPTION = {"photo": "Perception : capture caméra + encodage", "voix": "Perception : reconnaissance vocale (STT)"}
ETAPES = {"perception": "", "inference": "Inférence : modèle de décision ou LLM", "action": "Action : réponse vocale (TTS)"}
# Palette validée (contraste, daltonisme) : une seule couleur d'accent, l'inférence ; le reste en gris neutres.
COULEURS_ETAPES = {"perception": "#8f8e88", "inference": "#2a78d6", "action": "#dcdbd6"}
ACCENT = COULEURS_ETAPES["inference"]
ENCRE, ENCRE_SECONDAIRE, ENCRE_DISCRETE, FOND, GRILLE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
SOUS_TITRES = {
    "Jev 1.14": "Modèle de décision · TypeSafe",
    "GPT-6 Luna Decisions": "Modèle de décision · OpenAI",
    "GPT-6 Luna (génératif)": "LLM génératif · OpenAI",
}
COLONNES = [
    "date",
    "mode",
    "photo",
    "commande",
    "entree",
    "attendu",
    "etat_photo",
    "cerveau",
    "verdict",
    "juste",
    "p_bon_produit",
    "etat",
    "confiance_etat",
    "doute",
    "preparation_ms",
    "latence_ms",
    "parole_s",
    "cout_usd",
    "tokens_entree",
    "erreur",
]
FICHIER_CSV = DOSSIER_RESULTATS / "banc.csv"
# Ce qui change d'un mode à l'autre sur les images LinkedIn.
TEXTES_LINKEDIN = {
    "photo": {"bandeau": "ROBOT REACHY MINI (POLLEN ROBOTICS)  ·  CONTRÔLE QUALITÉ PAR VISION", "fichier": "linkedin.png"},
    "voix": {"bandeau": "ROBOT REACHY MINI (POLLEN ROBOTICS)  ·  CONTRÔLE QUALITÉ À LA VOIX", "fichier": "linkedin_voix.png"},
}


# --- mesures -------------------------------------------------------------------------------------------------------


def charger_donnees(chemin: Path = DOSSIER_DONNEES / "controle.json") -> dict[str, Any]:
    return json.loads(chemin.read_text(encoding="utf-8"))


def _chronometrer_preparation(photos: list[str]) -> tuple[dict[str, str], dict[str, float]]:
    """Les images prêtes à envoyer, et le temps pris pour chacune (redimensionner + encoder), en ms."""
    preparer_image(DOSSIER_DONNEES / "photos" / photos[0])  # chauffe : chargement des bibliothèques d'image
    images, durees_ms = {}, {}
    for photo in photos:
        debut = time.perf_counter()
        images[photo] = preparer_image(DOSSIER_DONNEES / "photos" / photo)
        durees_ms[photo] = (time.perf_counter() - debut) * 1000
    return images, durees_ms


def mesurer(
    client: httpx.Client,
    donnees: dict[str, Any],
    cerveaux: list[str] | None = None,
    modes: tuple[str, ...] = tuple(MODES),
    transcriptions: dict[str, str] | None = None,
    afficher: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Une ligne par (mode, contrôle, modèle). En mode voix, les modèles reçoivent ce que le STT a compris
    (`transcriptions` : phrase prononcée → transcription), comme le recevrait le robot."""
    cerveaux = cerveaux or list(CERVEAUX)
    cas, descriptions = donnees["cas"], donnees["descriptions"]
    images, preparation_ms = _chronometrer_preparation(list(dict.fromkeys(c["photo"] for c in cas)))
    parole_s = voix.durees(PHRASES)
    for nom in cerveaux:  # chauffe : connexion ouverte, rien de compté
        controler(client, nom, cas[0]["commande"], image_url=images[cas[0]["photo"]])
    jour = time.strftime("%Y-%m-%d")
    lignes = []
    for mode in modes:
        for i, c in enumerate(cas):
            phrase = (transcriptions or {}).get(descriptions[c["photo"]], descriptions[c["photo"]])
            entree = {"image_url": images[c["photo"]]} if mode == "photo" else {"description": phrase}
            decalage = i % len(cerveaux)
            for nom in cerveaux[decalage:] + cerveaux[:decalage]:
                resultat = controler(client, nom, c["commande"], **entree)
                lignes.append(
                    {
                        **c,
                        "date": jour,
                        "mode": mode,
                        "entree": c["photo"] if mode == "photo" else phrase,
                        **resultat.model_dump(),
                        "juste": resultat.verdict == c["attendu"],
                        "preparation_ms": preparation_ms[c["photo"]] if mode == "photo" else 0.0,
                        "parole_s": parole_s.get(reaction_pour(resultat)),
                    }
                )
                afficher(
                    f"{mode:<6} {i + 1:>2}/{len(cas)}  {nom:<24} {resultat.verdict or 'ERREUR':<16} attendu {c['attendu']:<16} "
                    f"{resultat.latence_ms:6.0f} ms  {resultat.erreur or ''}"
                )
    return lignes


def charger_lignes(chemin: Path = FICHIER_CSV) -> list[dict[str, Any]]:
    """Relit banc.csv avec les bons types : on peut refaire tableaux et graphiques sans repayer d'appels."""
    flottants = ("latence_ms", "preparation_ms", "parole_s", "cout_usd", "p_bon_produit", "confiance_etat")
    lignes = []
    with chemin.open(encoding="utf-8") as fichier:
        for ligne in csv.DictReader(fichier):
            for cle in flottants:
                ligne[cle] = float(ligne[cle]) if ligne.get(cle) else None
            ligne["tokens_entree"] = int(ligne["tokens_entree"]) if ligne.get("tokens_entree") else None
            ligne["juste"], ligne["doute"] = ligne["juste"] == "True", ligne["doute"] == "True"
            ligne["erreur"] = ligne["erreur"] or None
            ligne.setdefault("entree", ligne["photo"] if ligne["mode"] == "photo" else "")
            lignes.append(ligne)
    return lignes


# --- résumé ---------------------------------------------------------------------------------------------------------


def _moyenne(valeurs: list[float]) -> float | None:
    return statistics.fmean(valeurs) if valeurs else None


def _mediane(valeurs: list[float]) -> float | None:
    return statistics.median(valeurs) if valeurs else None


def resumer(
    lignes: list[dict[str, Any]],
    cerveaux: list[str] | None = None,
    modes: list[str] | None = None,
    perception: dict[str, float | None] | None = None,
) -> list[dict[str, Any]]:
    """Les chiffres par mode et par modèle. L'exactitude compte les erreurs réseau comme fausses.
    `perception` : durée mesurée à part pour chaque mode, en secondes (capture + encodage, ou STT)."""
    resume = []
    for mode in modes or [m for m in MODES if any(ligne["mode"] == m for ligne in lignes)]:
        for nom in cerveaux or list(CERVEAUX):
            siennes = [ligne for ligne in lignes if ligne["mode"] == mode and ligne["cerveau"] == nom]
            valides = [ligne for ligne in siennes if not ligne["erreur"]]
            latences = [ligne["latence_ms"] for ligne in valides]
            cout = _moyenne([ligne["cout_usd"] for ligne in valides if ligne["cout_usd"] is not None])
            mesuree = (perception or {}).get(mode)
            etapes = {
                "perception": mesuree if mesuree is not None else (_mediane([ligne["preparation_ms"] for ligne in valides]) or 0.0) / 1000,
                "inference": (_mediane(latences) or 0.0) / 1000,
                "action": _mediane([ligne["parole_s"] for ligne in valides if ligne["parole_s"] is not None]) or 0.0,
            }
            resume.append(
                {
                    "mode": mode,
                    "cerveau": nom,
                    "controles": len(siennes),
                    "erreurs": len(siennes) - len(valides),
                    "justes": sum(ligne["juste"] for ligne in siennes),
                    "produit_juste": sum((ligne["p_bon_produit"] >= 0.5) == (ligne["attendu"] != "mauvais_produit") for ligne in valides),
                    "etat_juste": sum(ligne["etat"] == ligne["etat_photo"] for ligne in valides),
                    "doutes": sum(ligne["doute"] for ligne in valides),
                    "latence_mediane_ms": _mediane(latences),
                    "latence_max_ms": max(latences) if latences else None,
                    "etapes_s": etapes,
                    "total_s": sum(etapes.values()),
                    "cout_moyen_usd": cout,
                    "cout_million_usd": cout * 1_000_000 if cout is not None else None,
                    "tokens_moyens": _moyenne([ligne["tokens_entree"] for ligne in valides if ligne["tokens_entree"] is not None]),
                    "source": "experimentation",
                }
            )
    return resume


def perception_mesuree(capture: dict[str, Any] | None, stt: dict[str, Any] | None) -> dict[str, float]:
    """La perception mesurée à part, en secondes : capture + encodage sur le robot (vision), transcription (voix)."""
    perception = {}
    if capture:
        perception["photo"] = (capture["capture_mediane_ms"] + capture["preparation_mediane_ms"]) / 1000
    if stt and stt.get("latence_mediane_ms"):
        perception["voix"] = stt["latence_mediane_ms"] / 1000
    return perception


def note_transcription(lignes: list[dict[str, Any]], stt: dict[str, Any] | None) -> str | None:
    """Combien d'erreurs du mode voix portent sur une phrase mal transcrite : sépare l'oreille (STT) du modèle."""
    erreurs = [ligne for ligne in lignes if ligne["mode"] == "voix" and not ligne["juste"]]
    if not stt or not erreurs:
        return None
    fausses = {ligne["transcription"] for ligne in stt["phrases"] if ligne["transcription"] and ligne["similarite"] < SEUIL_EXACTE}
    dues = sum(ligne["entree"] in fausses for ligne in erreurs)
    if dues == len(erreurs):
        return f"toutes les erreurs ({dues}) portent sur une phrase mal transcrite par la STT"
    return f"{dues} erreurs sur {len(erreurs)} portent sur une phrase mal transcrite par la STT"


def remplacer_par_robot(resume: list[dict[str, Any]], bout_en_bout: dict[str, Any]) -> list[dict[str, Any]]:
    """Pour la vision, les latences et coûts mesurés de bout en bout sur le robot (caméra réelle) remplacent ceux de
    l'expérimentation ; l'exactitude reste celle des 16 contrôles étiquetés."""
    nouveau = []
    for r in resume:
        m = bout_en_bout["modeles"].get(r["cerveau"]) if r["mode"] == "photo" else None
        if m is None or m.get("inference_mediane_ms") is None:
            nouveau.append(r)
            continue
        etapes = {
            "perception": (m["capture_mediane_ms"] + m["encodage_mediane_ms"]) / 1000,
            "inference": m["inference_mediane_ms"] / 1000,
            "action": m["reponse_vocale_mediane_s"],
        }
        cout = m.get("cout_moyen_usd")
        nouveau.append(
            {
                **r,
                "latence_mediane_ms": m["inference_mediane_ms"],
                "latence_max_ms": m["inference_max_ms"],
                "etapes_s": etapes,
                "total_s": sum(etapes.values()),
                "cout_moyen_usd": cout,
                "cout_million_usd": cout * 1_000_000 if cout is not None else None,
                "tokens_moyens": m.get("tokens_entree_median"),
                "source": "robot",
            }
        )
    return nouveau


def _nombre(valeur: float | None, decimales: int = 0) -> str:
    """Nombre à la française : espace pour les milliers, virgule décimale."""
    if valeur is None:
        return "–"
    return f"{valeur:,.{decimales}f}".replace(",", " ").replace(".", ",")


def _fois(rapport: float) -> str:
    """7,8 → « près de 8 fois » ; 7,2 → « plus de 7 fois » : jamais d'arrondi qui enjolive sans le dire."""
    entier = round(rapport)
    return f"près de {entier} fois" if entier > rapport else f"plus de {entier} fois"


def _duree(secondes: float) -> str:
    return f"{_nombre(secondes * 1000)} ms" if secondes < 1 else f"{_nombre(secondes, 1)} s"


def tableau_markdown(resume: list[dict[str, Any]], mode: str) -> str:
    lignes = [
        "| Modèle | Exactitude | Bon produit | État | Doutes | Inférence (médiane) | Pire inférence | Perception | Réponse vocale "
        "| Latence de bout en bout | Coût par requête | Coût par million | Tokens d'entrée |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in (r for r in resume if r["mode"] == mode):
        n = r["controles"]
        lignes.append(
            f"| {r['cerveau']} | **{r['justes']}/{n}** | {r['produit_juste']}/{n} | {r['etat_juste']}/{n} | {r['doutes']} "
            f"| **{_nombre(r['latence_mediane_ms'])} ms** | {_nombre(r['latence_max_ms'])} ms "
            f"| {_duree(r['etapes_s']['perception'])} | {_nombre(r['etapes_s']['action'], 1)} s "
            f"| **{_nombre(r['total_s'], 1)} s** | {_nombre(r['cout_moyen_usd'], 6)} $ | **{_nombre(r['cout_million_usd'])} $** "
            f"| {_nombre(r['tokens_moyens'])} |"
        )
    return "\n".join(lignes)


# --- graphiques ------------------------------------------------------------------------------------------------------


def _rc_police() -> dict[str, Any]:
    return {"font.family": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"]}


def _etiquette(ax: Any, gauche: float, y: float, largeur: float, texte: str, **style: Any) -> None:
    """La durée au centre du segment, seulement si elle y tient avec une marge ; sinon la légende la donne déjà."""
    etiquette = ax.text(gauche + largeur / 2, y, texte, ha="center", va="center", zorder=3, **style)
    place = ax.transData.transform((gauche + largeur, y))[0] - ax.transData.transform((gauche, y))[0]
    if etiquette.get_window_extent(ax.figure.canvas.get_renderer()).width > place - 14:
        etiquette.remove()


def libelles_perception(capture: dict[str, Any] | None, stt: dict[str, Any] | None, bout_en_bout: dict[str, Any] | None) -> dict[str, str]:
    """La légende de la perception, avec ses durées mesurées : « capture 13 ms + encodage 11 ms », « STT 0,97 s »."""
    libelles = dict(PERCEPTION)
    if bout_en_bout:
        modeles = [m for m in bout_en_bout["modeles"].values() if m["capture_mediane_ms"] is not None]
        capture_ms = statistics.median(m["capture_mediane_ms"] for m in modeles)
        encodage_ms = statistics.median(m["encodage_mediane_ms"] for m in modeles)
        libelles["photo"] = f"Perception : capture {_nombre(capture_ms)} ms + encodage {_nombre(encodage_ms)} ms"
    elif capture:
        libelles["photo"] = (
            f"Perception : capture {_nombre(capture['capture_mediane_ms'])} ms + encodage {_nombre(capture['preparation_mediane_ms'])} ms"
        )
    if stt and stt.get("latence_mediane_ms"):
        libelles["voix"] = f"Perception : reconnaissance vocale (STT) {_nombre(stt['latence_mediane_ms'] / 1000, 2)} s"
    return libelles


def sources(stt: dict[str, Any] | None, bout_en_bout: dict[str, Any] | None) -> dict[str, str]:
    """Le pied de chaque image : « Expérimentations avec OpenRouter », sans date, et comment chaque étape a été mesurée.
    Clés : un pied par mode, et « tout » pour le graphique du README qui montre les deux."""
    if bout_en_bout:
        cycles = next(iter(bout_en_bout["modeles"].values()))["cycles"]
        photo = f"pipeline mesuré sur le Reachy Mini face à une vraie canette, médiane de {cycles} cycles par modèle"
        photo += " ; exactitude sur 16 contrôles"
    else:
        photo = "appels séquentiels, même image pour les trois modèles ; capture caméra et encodage mesurés sur le Reachy Mini"
    if stt:
        voix_ = f"voix du préparateur synthétisée puis transcrite par {stt['modele'].split('/')[-1]} (STT) ; appels séquentiels"
    else:
        voix_ = "appels séquentiels, même phrase pour les trois modèles"
    tts = "TTS : voix macOS « Thomas »."
    return {
        "photo": f"Source : Expérimentations avec OpenRouter ({photo}). {tts}",
        "voix": f"Source : Expérimentations avec OpenRouter ({voix_}). {tts}",
        "tout": f"Source : Expérimentations avec OpenRouter. {tts}\nVision : {photo}.\nVoix : {voix_}.",
    }


def graphique(resume: list[dict[str, Any]], chemin: Path, perception: dict[str, str], source: str) -> None:
    """Le détail pour le README, en paysage : un panneau par mode, une barre par modèle découpée en trois étapes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    modes = list(dict.fromkeys(r["mode"] for r in resume))
    cerveaux = list(dict.fromkeys(r["cerveau"] for r in resume))
    with plt.rc_context(_rc_police()):
        fig, axes = plt.subplots(1, len(modes), figsize=(13, 6.6), dpi=150, sharex=True, squeeze=False)
        fig.patch.set_facecolor(FOND)
        fig.subplots_adjust(left=0.15, right=0.98, top=0.74, bottom=0.2, wspace=0.08)
        x_max = max(r["total_s"] for r in resume) * 1.4
        for ax, mode in zip(axes[0], modes, strict=True):
            ax.set_facecolor(FOND)
            ax.set_xlim(0, x_max)
            for i, nom in enumerate(cerveaux):
                r = next(r for r in resume if r["mode"] == mode and r["cerveau"] == nom)
                y, gauche = len(cerveaux) - 1 - i, 0.0
                for etape, couleur in COULEURS_ETAPES.items():
                    largeur = r["etapes_s"][etape]
                    if largeur <= 0:
                        continue
                    ax.barh(y, largeur, left=gauche, height=0.56, color=couleur, edgecolor=FOND, linewidth=2, zorder=2)
                    _etiquette(
                        ax,
                        gauche,
                        y,
                        largeur,
                        f"{_nombre(largeur, 2 if largeur < 1 else 1)} s",
                        fontsize=10.5,
                        fontweight="bold",
                        color="white" if etape == "inference" else ENCRE,
                    )
                    gauche += largeur
                inference = r["etapes_s"]["inference"]
                ax.text(
                    gauche + x_max * 0.015,
                    y + 0.12,
                    f"{_nombre(r['total_s'], 1)} s",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color=ENCRE,
                )
                ax.text(
                    gauche + x_max * 0.015,
                    y - 0.22,
                    f"dont inférence {_nombre(inference, 2 if inference < 1 else 1)} s\nexactitude {r['justes']}/{r['controles']}",
                    va="center",
                    fontsize=9,
                    color=ENCRE_SECONDAIRE,
                    linespacing=1.3,
                )
            ax.set_title(MODES[mode], loc="left", fontsize=12.5, fontweight="bold", color=ENCRE, pad=10)
            ax.set_yticks(
                range(len(cerveaux) - 1, -1, -1), cerveaux if ax is axes[0][0] else [""] * len(cerveaux), fontsize=11.5, color=ENCRE
            )
            ax.tick_params(axis="y", length=0, pad=8)
            ax.tick_params(axis="x", colors=ENCRE_SECONDAIRE, labelsize=10)
            ax.set_xlabel("latence (s)", color=ENCRE_SECONDAIRE, fontsize=10)
            ax.grid(axis="x", color=GRILLE, linewidth=1, zorder=0)
            for cote in ("top", "right", "left"):
                ax.spines[cote].set_visible(False)
            ax.spines["bottom"].set_color(GRILLE)
        legende = [Patch(color=COULEURS_ETAPES["perception"], label=f"{perception[m]} ({MODES[m].split(' (')[0].lower()})") for m in modes]
        legende += [Patch(color=COULEURS_ETAPES[e], label=ETAPES[e]) for e in ("inference", "action")]
        fig.legend(handles=legende, loc="upper left", bbox_to_anchor=(0.03, 0.905), ncol=2, frameon=False, fontsize=10, labelcolor=ENCRE)
        fig.text(
            0.03, 0.96, "Latence de bout en bout de Reachy : perception, inférence, action", fontsize=18, fontweight="bold", color=ENCRE
        )
        fig.text(0.03, 0.02, source, fontsize=8, color=ENCRE_SECONDAIRE, linespacing=1.5)
        fig.savefig(chemin, facecolor=FOND)
        plt.close(fig)


def graphique_linkedin(
    resume: list[dict[str, Any]], chemin: Path, mode: str, perception: str, source: str, note: str | None = None
) -> None:
    """L'image du post LinkedIn (1 920 × 1 080) : un message, une couleur d'accent (l'inférence), le coût en colonne,
    l'exactitude en colonne quand elle diffère d'un modèle à l'autre (`note` dit alors d'où viennent les erreurs)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    lignes = [r for r in resume if r["mode"] == mode]
    type_de = {r["cerveau"]: CERVEAUX.get(r["cerveau"], {}).get("type") for r in lignes}
    rapides = [r["etapes_s"]["inference"] for r in lignes if type_de[r["cerveau"]] == "decision"]
    generatives = [r["etapes_s"]["inference"] for r in lignes if type_de[r["cerveau"]] == "llm"]
    titre = "Modèles de décision : l'inférence qui fait réagir le robot"
    if rapides and generatives:
        titre = f"Modèle de décision : une inférence {_fois(statistics.fmean(generatives) / statistics.fmean(rapides))} plus rapide"
    exacts = all(r["justes"] == r["controles"] for r in lignes)
    n = lignes[0]["controles"]

    with plt.rc_context(_rc_police()):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes((0.29, 0.18, 0.55 if exacts else 0.45, 0.47))
        x_max = max(r["total_s"] for r in lignes) * (1.38 if exacts else 1.45)  # de la place pour les étiquettes en bout de barre
        ax.set_xlim(0, x_max)  # avant le dessin : la colonne de coût se place avec ces limites
        ax.set_ylim(-0.6, len(lignes) - 0.4)
        for i, r in enumerate(lignes):
            y, gauche = len(lignes) - 1 - i, 0.0
            for etape, couleur in COULEURS_ETAPES.items():
                largeur = r["etapes_s"][etape]
                if largeur <= 0:
                    continue
                ax.barh(y, largeur, left=gauche, height=0.56, color=couleur, edgecolor="white", linewidth=2.5, zorder=2)
                _etiquette(
                    ax,
                    gauche,
                    y,
                    largeur,
                    f"{_nombre(largeur, 2 if largeur < 1 else 1)} s",
                    fontsize=14 if etape == "inference" else 12,
                    fontweight="bold" if etape == "inference" else "normal",
                    color={"perception": ENCRE, "inference": "white"}.get(etape, ENCRE_SECONDAIRE),
                )
                gauche += largeur
            inference = r["etapes_s"]["inference"]
            ax.text(
                r["total_s"] + x_max * 0.02,
                y + 0.11,
                f"{_nombre(r['total_s'], 1)} s",
                ha="left",
                va="center",
                fontsize=19,
                fontweight="bold",
                color=ENCRE,
            )
            detail = f"dont {_nombre(inference, 2 if inference < 1 else 1)} s d'inférence"
            ax.text(
                r["total_s"] + x_max * 0.02,
                y - 0.2,
                detail,
                ha="left",
                va="center",
                fontsize=10.5,
                color=ACCENT if inference < 1 else ENCRE_SECONDAIRE,
            )
            ax.text(
                -0.04,
                y + 0.12,
                r["cerveau"].replace(" (génératif)", ""),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=ENCRE,
            )
            ax.text(
                -0.04,
                y - 0.17,
                SOUS_TITRES.get(r["cerveau"], ""),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=11,
                color=ENCRE_SECONDAIRE,
            )
            hauteur = ax.transData.transform((0, y))[1] / fig.bbox.height
            if not exacts:
                fig.text(
                    0.80, hauteur, f"{r['justes']}/{r['controles']}", ha="center", va="center", fontsize=17, fontweight="bold", color=ENCRE
                )
            fig.text(
                0.925,
                hauteur,
                f"{_nombre(r['cout_million_usd'])} $",
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color=ENCRE,
            )
        ax.set_yticks([])
        graduations = range(int(max(r["total_s"] for r in lignes)) + 2)
        ax.set_xticks(graduations, [f"{s} s" for s in graduations])
        ax.tick_params(axis="x", length=0, colors=ENCRE_DISCRETE, labelsize=10.5, pad=8)
        ax.grid(axis="x", color="#eeede9", linewidth=1, zorder=0)
        for cote in ax.spines.values():
            cote.set_visible(False)

        fig.text(
            0.925,
            0.665,
            "Coût pour 1 million\nd'inférences",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=ENCRE_SECONDAIRE,
            linespacing=1.3,
        )
        if not exacts:
            fig.text(
                0.80,
                0.665,
                f"Exactitude\n(sur {n} contrôles)",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color=ENCRE_SECONDAIRE,
                linespacing=1.3,
            )
        fig.text(0.05, 0.935, TEXTES_LINKEDIN[mode]["bandeau"], fontsize=11, fontweight="bold", color=ACCENT)
        fig.text(0.05, 0.868, titre, fontsize=25, fontweight="bold", color=ENCRE)
        fig.text(
            0.05,
            0.818,
            "Pipeline perception → inférence → action · latence médiane"
            + (f" · exactitude {n}/{n} pour les trois modèles" if exacts else f" · {note}" if note else ""),
            fontsize=12.5,
            color=ENCRE_SECONDAIRE,
        )
        x = 0.05
        for etape, libelle in (("perception", perception), ("inference", ETAPES["inference"]), ("action", ETAPES["action"])):
            fig.patches.append(Rectangle((x, 0.748), 0.014, 0.025, transform=fig.transFigure, color=COULEURS_ETAPES[etape]))
            fig.text(x + 0.021, 0.760, libelle, va="center", fontsize=10.5, color=ENCRE)
            x += 0.021 + len(libelle) * 0.0064 + 0.025
        fig.lines.append(Line2D([0.05, 0.95], [0.115, 0.115], transform=fig.transFigure, color=GRILLE, linewidth=1))
        fig.text(0.05, 0.07, source, fontsize=8.5, color=ENCRE_DISCRETE)
        fig.savefig(chemin, facecolor="white")
        plt.close(fig)


# --- écriture -------------------------------------------------------------------------------------------------------


def ecrire_resultats(
    lignes: list[dict[str, Any]],
    resume: list[dict[str, Any]],
    dossier: Path = DOSSIER_RESULTATS,
    perception: dict[str, str] | None = None,
    pieds: dict[str, str] | None = None,
    notes: dict[str, str | None] | None = None,
) -> list[Path]:
    """banc.csv (toutes les mesures), resume.md (un tableau par mode), latence.png (le détail, pour le README),
    linkedin.png et linkedin_voix.png (les images du post)."""
    perception = perception or dict(PERCEPTION)
    pieds = pieds or sources(None, None)
    notes = notes or {}
    dossier.mkdir(parents=True, exist_ok=True)
    chemin_csv, chemin_md, chemin_png = dossier / "banc.csv", dossier / "resume.md", dossier / "latence.png"
    with chemin_csv.open("w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=COLONNES, extrasaction="ignore")
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    modes = list(dict.fromkeys(r["mode"] for r in resume))
    modeles = ", ".join(f"`{config['model']}`" for config in CERVEAUX.values())
    sections = "\n\n".join(
        f"## {MODES[mode]}\n\n{perception[mode]}.\n\n{tableau_markdown(resume, mode)}"
        + (f"\n\nExactitude : {notes[mode]}." if notes.get(mode) else "")
        for mode in modes
    )
    chemin_md.write_text(
        f"# Résultats\n\nGénéré par `uv run reachy-jev banc`. Modèles : {modeles}.\n\n{sections}\n\n"
        + "\n\n".join(f"*{MODES[m]} : {pieds[m]}*" for m in modes)
        + "\n",
        encoding="utf-8",
    )
    graphique(resume, chemin_png, perception, pieds["tout"])
    chemins = [chemin_csv, chemin_md, chemin_png]
    for mode in modes:
        chemin = dossier / TEXTES_LINKEDIN[mode]["fichier"]
        graphique_linkedin(resume, chemin, mode, perception[mode], pieds[mode], notes.get(mode))
        chemins.append(chemin)
    return chemins
