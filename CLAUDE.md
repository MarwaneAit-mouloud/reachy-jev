# reachy-jev

## Ce que c'est
Un Reachy Mini Wireless (réseau) qui contrôle des commandes : il lit la ligne de commande (texte) et regarde l'article (photo),
puis hoche la tête, la secoue, fait la moue ou demande de remontrer l'article, en parlant. Trois IA comparées sur la même décision :
Jev 1.14 (TypeSafe), GPT-6 Luna Decisions (OpenAI), GPT-6 Luna génératif. Sert un repo GitHub et UN post LinkedIn sur la latence
et le coût de l'IA en robotique (photo du robot + graphique + texte, pas de vidéo).

## Rester léger
C'est un post, pas une étude. Trois commandes : `banc` (`--redessiner` pour refaire tableaux et graphiques sans appel),
`capture` (perception mesurée sur le vrai robot) et `demo`. Pas de nouveau cas d'usage ni d'analyse en plus sans demande.
Le user apprend Jev lui-même dans `notebooks/` : ses notebooks, exclus du dépôt (`.gitignore`), dépendances dans le groupe
`notebook` (`uv sync --group notebook`). N'y écrire que s'il le demande. Anciennes versions (tri des déchets) : `../_sauvegardes/`.

## Git
Commits en gitmoji (🎉 ✨ 🐛 🎨 🔧 🙈 🍱 📝 ✅…), titre à l'impératif en français, corps qui explique le pourquoi.
Avant tout commit : `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run pytest`. Jamais de clé ni de `.env`.

## Honnêteté
- Aucun chiffre écrit à la main dans les résultats : `banc` écrit `resultats/` (csv, resume.md, latence.png). Un résultat défavorable à Jev se publie tel quel.
- Les annonces de TypeSafe sont présentées comme les leurs.
- Les photos de `data/photos/` viennent de Wikimedia Commons : crédits et licences dans `data/photos/CREDITS.md`, à garder à jour.
- README court (cas d'usage, un tableau, cinq points à retenir, commandes) ; tout le détail va dans `docs/ANALYSE.md`.
  Les chiffres recopiés dans ces deux fichiers doivent rester ceux de `resultats/`.

## Code
- Python 3.11, `uv` (jamais `pip`), Pydantic, SDK `reachy-mini` 1.8.x. Commentaires et noms en français.
- `controle.py` = le cas d'usage : deux questions (Noul « bon produit ? », Choice « état ») et le code tranche le verdict.
  Deux modes : la vision (photo), ou la voix (phrase du préparateur transcrite par la STT de `stt.py`, sans photo).
- `banc.py` détaille la latence de bout en bout en trois étapes : perception (vision : capture + encodage mesurés sur le
  robot, `resultats/capture.json` ; voix : STT, `resultats/stt.json`), inférence (appel au modèle), action (réponse vocale TTS).
  Si `resultats/bout_en_bout.json` existe (`camera.mesurer_bout_en_bout`, vrai robot), ses latences et coûts remplacent
  ceux de la vision. Le mode voix décide sur les transcriptions de `stt.json`, pas sur les phrases exactes.
  Images : latence.png (détail, deux panneaux, pour le README), linkedin.png et linkedin_voix.png (1 920 × 1 080, un seul
  accent, police Avenir Next, vocabulaire IA/robotique). Pied des images : « Expérimentations avec OpenRouter », sans date. Ne pas réécrire « c'est un signal, pas une étude » : le user n'aime pas.
- `camera.py` n'enregistre jamais d'image (vie privée) : seulement des durées.
- Les modèles de décision passent par l'API Decisions en HTTP (`openrouter.py`), PAS par le SDK `typesafe_sdk` :
  via OpenRouter, le SDK envoie les images en multipart, refusé (400).
- Images : top-level du `state` (seul endroit lu par tous les modèles), JPEG ≤ 1 024 px, la même pour tous.
- Aucune exception réseau ne sort d'un appel : un échec est un `Controle` avec `erreur`.
- Tests : réseau remplacé par `httpx.MockTransport`, robot par un objet de test local, jamais en vrai. Ne jamais patcher `time.sleep` globalement.
- Robot : `enable_motors()` puis `wake_up()` ; `goto_target` recentre le corps si `body_yaw` n'est pas `None` ;
  parole poussée en flux (`push_audio_sample`), pas `play_sound` (~3 s de transfert sur ce robot). Endormir en fin de démo.
