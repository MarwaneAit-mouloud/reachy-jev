"""Les commandes du projet :

uv run reachy-jev banc                                                 # data/controle.json × 3 modèles, vision puis voix → resultats/
uv run reachy-jev banc --mode voix                                     # ne relance qu'un mode, garde les mesures de l'autre
uv run reachy-jev banc --redessiner                                    # refait tableaux et graphiques depuis banc.csv, sans appel
uv run reachy-jev capture                                              # mesure la perception sur le vrai Reachy (caméra)
uv run reachy-jev bout-en-bout --commande "1 canette de San Pellegrino"  # pipeline complet sur Reachy, pour chaque modèle
uv run reachy-jev stt                                                  # mesure la reconnaissance vocale du contrôle à la voix
uv run reachy-jev demo                                                 # contrôle en direct : caméra, voix et gestes de Reachy
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from reachy_jev import camera, stt
from reachy_jev import voix as voix_robot
from reachy_jev.banc import (
    FICHIER_CSV,
    MODES,
    charger_donnees,
    charger_lignes,
    ecrire_resultats,
    libelles_perception,
    mesurer,
    note_transcription,
    perception_mesuree,
    remplacer_par_robot,
    resumer,
    sources,
    tableau_markdown,
)
from reachy_jev.controle import VERDICTS, Controle, controler, preparer_image
from reachy_jev.openrouter import creer_client
from reachy_jev.reglages import CERVEAU_PILOTE, CERVEAUX, hote_robot
from reachy_jev.robot import PHRASES, Reachy, reaction_pour, reagir


def afficher(controle: Controle) -> None:
    if controle.erreur:
        print(f"   {controle.cerveau:<24} ERREUR : {controle.erreur}")
        return
    doute = "  (doute)" if controle.doute else ""
    cout = f"{controle.cout_usd:.6f} $" if controle.cout_usd is not None else ""
    print(f"   {controle.cerveau:<24} {VERDICTS[controle.verdict]:<24}{doute:<10} {controle.latence_ms:6.0f} ms   {cout}")


def controler_en_direct(
    client: httpx.Client, robot: Reachy, commande: str, voix: dict[str, Path | None] | None = None
) -> dict[str, Controle]:
    """Perception, puis les trois modèles en parallèle ; le robot parle et bouge dès que le pilote (Jev) a répondu."""
    debut = time.perf_counter()
    image_url = preparer_image(robot.photo())
    print(f"   📷 Perception (capture + encodage) : {(time.perf_counter() - debut) * 1000:.0f} ms")
    resultats: dict[str, Controle] = {}
    geste: threading.Thread | None = None
    with ThreadPoolExecutor(max_workers=len(CERVEAUX)) as pool:
        futurs = [pool.submit(controler, client, nom, commande, image_url) for nom in CERVEAUX]
        for futur in as_completed(futurs):
            controle = futur.result()
            resultats[controle.cerveau] = controle
            afficher(controle)
            if controle.cerveau == CERVEAU_PILOTE:
                print(f"   🤖 Reachy : « {PHRASES[reaction_pour(controle)]} »")
                geste = threading.Thread(target=reagir, args=(robot, controle, voix))
                geste.start()
    if geste is not None:
        geste.join()
    return resultats


def lancer_banc(redessiner: bool, modes: list[str]) -> None:
    """Le banc (appels payants) sur les modes demandés, en gardant les mesures des autres ; ou seulement les tableaux et
    graphiques à partir de banc.csv. Intègre les mesures faites à part : capture, bout en bout sur le robot, STT."""
    capture, bout_en_bout, mesure_stt = camera.lire(), camera.lire(camera.FICHIER_BOUT_EN_BOUT), stt.lire()
    anciennes = [ligne for ligne in charger_lignes() if ligne["mode"] in MODES] if FICHIER_CSV.exists() else []
    if redessiner:
        lignes = anciennes
    else:
        if "voix" in modes and mesure_stt is None:
            raise SystemExit(
                "Mode voix : lance d'abord `uv run reachy-jev stt`, le banc décide sur ce que la reconnaissance vocale a compris."
            )
        transcriptions = {
            ligne["texte"]: ligne["transcription"] for ligne in (mesure_stt or {"phrases": []})["phrases"] if ligne["transcription"]
        }
        donnees = charger_donnees()
        print(
            f"{len(donnees['cas'])} contrôles × {len(modes)} mode(s) ({', '.join(modes)}) × {len(CERVEAUX)} modèles, "
            "appels un par un (~2 min par mode, moins d'un centime)…\n"
        )
        nouvelles = mesurer(creer_client(), donnees, modes=tuple(modes), transcriptions=transcriptions)
        lignes = [ligne for ligne in anciennes if ligne["mode"] not in modes] + nouvelles
        lignes.sort(key=lambda ligne: list(MODES).index(ligne["mode"]))  # tri stable : l'ordre des mesures est gardé
    resume = resumer(lignes, perception=perception_mesuree(capture, mesure_stt))
    if bout_en_bout:
        resume = remplacer_par_robot(resume, bout_en_bout)
    chemins = ecrire_resultats(
        lignes,
        resume,
        perception=libelles_perception(capture, mesure_stt, bout_en_bout),
        pieds=sources(mesure_stt, bout_en_bout),
        notes={"voix": note_transcription(lignes, mesure_stt)},
    )
    for mode in dict.fromkeys(r["mode"] for r in resume):
        print(f"\n{MODES[mode]}\n{tableau_markdown(resume, mode)}")
    if capture is None and bout_en_bout is None:
        print("\nAucune mesure sur le robot : lance `uv run reachy-jev bout-en-bout` avec le robot allumé pour la perception réelle.")
    print("\nÉcrit : " + ", ".join(str(chemin) for chemin in chemins))


def lancer_capture(images: int) -> None:
    """Mesure la perception sur le vrai robot : première image, puis des prises de vue neuves. N'enregistre aucune image."""
    robot = Reachy(hote_robot())
    print(robot.ouvrir())
    try:
        mesure = camera.mesurer(robot, images)
    finally:
        robot.dormir()
        robot.fermer()
    chemin = camera.ecrire(mesure)
    print(f"Première image après connexion : {mesure['premiere_image_ms']:.0f} ms")
    print(
        f"Image neuve : {mesure['capture_mediane_ms']:.0f} ms en médiane (pire : {mesure['capture_max_ms']:.0f} ms), "
        f"encodage : {mesure['preparation_mediane_ms']:.0f} ms, résolution {mesure['resolution'][0]} × {mesure['resolution'][1]}"
    )
    print(f"Écrit : {chemin}. Lance `uv run reachy-jev banc --redessiner` pour l'intégrer aux graphiques.")


def lancer_bout_en_bout(cycles: int, commande: str) -> None:
    """Le pipeline complet sur le vrai robot, pour chaque modèle : capture → encodage → inférence → réponse vocale."""
    client = creer_client()
    robot = Reachy(hote_robot())
    print(robot.ouvrir())
    print(f"« {commande} » : {cycles} cycles × {len(CERVEAUX)} modèles, prises de vue réelles, appels un par un…")
    try:
        mesure = camera.mesurer_bout_en_bout(robot, client, commande, cycles)
    finally:
        robot.dormir()
        robot.fermer()
    chemin = camera.ecrire(mesure, camera.FICHIER_BOUT_EN_BOUT)
    for nom, m in mesure["modeles"].items():
        print(
            f"  {nom:<24} capture {m['capture_mediane_ms']:.0f} ms · encodage {m['encodage_mediane_ms']:.0f} ms · "
            f"inférence {m['inference_mediane_ms']:.0f} ms · voix {m['reponse_vocale_mediane_s']:.1f} s · "
            f"total {m['total_median_s']:.2f} s · {m['tokens_entree_median']:.0f} tokens · réactions {m['reactions']}"
        )
    print(f"Écrit : {chemin}. Lance `uv run reachy-jev banc --redessiner` pour l'intégrer aux graphiques.")


def lancer_stt() -> None:
    """Mesure la perception du contrôle à la voix : la phrase du préparateur, synthétisée, puis transcrite (STT)."""
    phrases = list(dict.fromkeys(charger_donnees()["descriptions"].values()))
    mesure = stt.mesurer(creer_client(), phrases)
    chemin = stt.ecrire(mesure)
    for ligne in mesure["phrases"]:
        print(f"  {ligne['latence_ms']:5d} ms  {ligne['similarite']:.2f}  « {ligne['transcription']} »")
    print(
        f"Transcription : {mesure['latence_mediane_ms']} ms en médiane pour {mesure['duree_audio_mediane_s']} s de parole, "
        f"{mesure['transcriptions_exactes']}/{len(mesure['phrases'])} exactes ({mesure['modele']})."
    )
    print(f"Écrit : {chemin}. Lance `uv run reachy-jev banc --redessiner` pour l'intégrer aux graphiques.")


def lancer_demo() -> None:
    """Contrôle en direct : on tape la ligne de commande, on montre l'article à la caméra de Reachy, il réagit."""
    client = creer_client()
    voix = voix_robot.preparer(PHRASES)
    print(f"Voix prête ({voix_robot.moteur()}) : {sum(v is not None for v in voix.values())}/{len(PHRASES)} phrases.")
    robot = Reachy(hote_robot())
    print(robot.ouvrir())
    print("Tape la ligne de commande (ex. « 1 canette de San Pellegrino »), montre l'article, Entrée. Ligne vide pour finir.")
    try:
        while commande := input("\n📦 ").strip():
            controler_en_direct(client, robot, commande, voix)
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        robot.dormir()
        robot.fermer()


def main() -> None:
    parseur = argparse.ArgumentParser(
        prog="reachy-jev", description="Reachy contrôle des commandes : Jev contre OpenAI Decisions et un LLM génératif."
    )
    sous = parseur.add_subparsers(dest="action", required=True)
    banc = sous.add_parser("banc", help="passe data/controle.json aux trois modèles, par la vision et par la voix, et écrit resultats/")
    banc.add_argument("--redessiner", action="store_true", help="refait tableaux et graphiques depuis resultats/banc.csv, sans appel")
    banc.add_argument(
        "--mode", choices=[*MODES, "tous"], default="tous", help="le mode à (re)mesurer ; les mesures des autres sont gardées"
    )
    capture = sous.add_parser("capture", help="mesure la perception (caméra + encodage) sur le vrai Reachy")
    capture.add_argument("--images", type=int, default=20, help="nombre de prises de vue neuves à chronométrer")
    bout = sous.add_parser("bout-en-bout", help="pipeline complet sur le vrai Reachy, pour chaque modèle")
    bout.add_argument("--cycles", type=int, default=8, help="nombre de cycles (une prise de vue et une inférence par modèle)")
    bout.add_argument("--commande", required=True, help="la ligne de commande contrôlée, à comparer à l'article montré à la caméra")
    sous.add_parser("stt", help="mesure la reconnaissance vocale du contrôle à la voix")
    sous.add_parser("demo", help="contrôle en direct ce qu'on montre à la caméra de Reachy")
    args = parseur.parse_args()

    if args.action == "banc":
        lancer_banc(args.redessiner, list(MODES) if args.mode == "tous" else [args.mode])
    elif args.action == "capture":
        lancer_capture(args.images)
    elif args.action == "bout-en-bout":
        lancer_bout_en_bout(args.cycles, args.commande)
    elif args.action == "stt":
        lancer_stt()
    else:
        lancer_demo()


if __name__ == "__main__":
    main()
