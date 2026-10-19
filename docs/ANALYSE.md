# Analyse détaillée

La méthode, chaque passage et les limites de l'expérimentation. Le résumé est dans le [README](../README.md).

## Le cas d'usage en détail

1. Le préparateur tape (ou scanne) la ligne de commande, par exemple « 1 canette de Coca-Cola ».
2. Il montre l'article à la caméra de Reachy.
3. Reachy envoie le texte et la photo à l'IA dans une seule requête, avec deux questions : *est-ce le bon produit ?* et *est-il abîmé ?* Le code tranche à partir des deux réponses.
4. Reachy parle et bouge en même temps :
   - bon produit, intact : il hoche la tête, « C'est bon, article validé ! » ;
   - mauvais produit : il secoue la tête, « Attention, ce n'est pas le bon produit. » ;
   - produit abîmé : il fait la moue, « Le produit est abîmé, on le met de côté. » ;
   - doute : il penche la tête, « Je ne suis pas sûr. Tu peux me le remontrer ? ».

La même photo est bonne pour une commande et mauvaise pour une autre : il faut lire le texte **et** voir l'image.

**Variante voix.** En entrepôt, on a souvent les mains prises : le préparateur décrit l'article à voix haute (« Un Coca-Cola, la canette est toute froissée. »), une reconnaissance vocale (STT) le transcrit, et les modèles décident sur cette transcription, sans photo. C'est aussi sur du texte que TypeSafe fait sa promesse de coût.

## Résultats

16 contrôles, sur 8 photos de canettes, chacune contrôlée avec la bonne commande puis avec une mauvaise. Le jeu contient des pièges : Sprite contre Diet Sierra Mist (deux sodas citron), Fanta contre Orangina, Coca contre Pepsi, canettes écrasées. Les appels passent un par un via OpenRouter, avec la même entrée pour les trois modèles. Exactitude par la vision mesurée sur ces photos le 17 octobre 2026 ; latences et coûts par la vision mesurés sur le robot, face à une vraie canette, le 19 octobre ; voix le 19 octobre.

Chaque barre du graphique détaille la latence de bout en bout du pipeline :
- **perception** : par la vision, capture d'une image neuve par la caméra du Reachy et encodage pour l'IA (mesurés sur le robot, voir plus bas) ; par la voix, la reconnaissance vocale de la phrase du préparateur ;
- **inférence** : l'appel au modèle de décision ou au LLM, réseau compris ;
- **action** : la réponse vocale de Reachy (TTS), synthétisée à l'avance, qui part dès que l'inférence est terminée.

### Par la vision (photo de l'article)

| Modèle | Exactitude | Inférence (médiane) | Pire inférence | Perception | Réponse vocale | Latence de bout en bout | Coût par million d'inférences | Tokens d'entrée |
|---|---|---|---|---|---|---|---|---|
| Jev 1.14 | **16/16** | **273 ms** | 441 ms | 22 ms | 1,8 s | **2,1 s** | **83 $** | 1 985 |
| GPT-6 Luna Decisions | **16/16** | **394 ms** | 544 ms | 22 ms | 1,8 s | **2,2 s** | **90 $** | 901 |
| GPT-6 Luna (génératif) | **16/16** | **6 153 ms** | 8 572 ms | 19 ms | 1,8 s | **8,0 s** | **116 $** | 882 |

Latences et coûts : pipeline complet sur le Reachy Mini, caméra pointée sur une canette de San Pellegrino, commande « 1 canette de San Pellegrino », 8 cycles par modèle (perception : capture 11 ms + encodage 11 ms). Les trois modèles l'ont validée 8 fois sur 8, d'où la réplique « C'est bon, article validé ! » (1,8 s). Exactitude : les 16 contrôles étiquetés.

Sur les photos du 17 octobre, plus grandes, l'inférence médiane était de 343, 328 et 2 606 ms, pour 97 $, 103 $ et 134 $ par million.

### Par la voix (le préparateur décrit l'article)

| Modèle | Exactitude | Inférence (médiane) | Pire inférence | Perception | Réponse vocale | Latence de bout en bout | Coût par million d'inférences | Tokens d'entrée |
|---|---|---|---|---|---|---|---|---|
| Jev 1.14 | **12/16** | **283 ms** | 488 ms | 0,97 s | 2,2 s | **3,4 s** | **17 $** | 405 |
| GPT-6 Luna Decisions | **13/16** | **283 ms** | 1 079 ms | 0,97 s | 2,2 s | **3,4 s** | **33 $** | 334 |
| GPT-6 Luna (génératif) | **14/16** | **4 447 ms** | 8 774 ms | 0,97 s | 2,2 s | **7,6 s** | **70 $** | 209 |

La voix du préparateur est synthétisée (voix macOS « Jacques », 2,9 s par phrase en médiane), puis transcrite par `google/gemini-2.5-flash-lite` via OpenRouter : 0,97 s en médiane, 32 $ par million de phrases. **Seules 4 phrases sur 8 sont transcrites exactement**, et certaines erreurs inversent le sens :

| Le préparateur dit | La STT comprend |
|---|---|
| « Un Coca-Cola, la canette est toute froissée. » | « Allez Coca-Cola, la canette est toute propre. » |
| « Une vieille canette de Pepsi, bien cabossée. » | « Une vieille canne de pépé, vieille comme le monde. » |
| « Une canette de Fanta Orange, ouverte mais pas déformée. » | « une canette de fontaine orange ouverte mais pas de forme » |
| « J'ai une canette de Diet Sierra Mist, impeccable. » | « J'ai une canette de diésel armiste, impeccable. » |

Les 9 erreurs des trois modèles portent toutes sur ces phrases mal transcrites. Sur les phrases exactes (passage du 17 octobre, sans STT), les trois modèles faisaient 16/16, en 225, 190 et 2 258 ms d'inférence médiane, pour 17 $, 33 $ et 47 $ par million.

Tableaux et graphiques sont générés par `uv run reachy-jev banc` (ou `banc --redessiner`, sans nouvel appel) : aucun chiffre n'est écrit à la main. Les images prêtes pour LinkedIn sont [`resultats/linkedin.png`](../resultats/linkedin.png) (vision) et [`resultats/linkedin_voix.png`](../resultats/linkedin_voix.png) (voix). Le détail des 96 mesures (exactitude sur chaque question, doutes, coût exact) est dans [`resultats/banc.csv`](../resultats/banc.csv) et [`resultats/resume.md`](../resultats/resume.md), celui des transcriptions dans [`resultats/stt.json`](../resultats/stt.json). Chaque passage a coûté moins d'un centime.

### Les prix de l'API (OpenRouter, 17 octobre 2026)

| Modèle | Entrée, par million de tokens | Sortie, par million de tokens |
|---|---|---|
| Jev 1.14 | 0,042 $ | gratuite |
| GPT-6 Luna Decisions | 0,10 $ | gratuite |
| GPT-6 Luna (génératif) | 0,10 $ en tarif standard (de 0,05 à 0,20 $ selon le fournisseur et le niveau de service) | 0,50 $ en tarif standard (de 0,25 à 1 $) |

OpenRouter choisit le fournisseur du modèle génératif à chaque appel ; les coûts mesurés sont ceux réellement facturés (`usage.cost`).

## Analyse

### 1. L'exactitude ne départage pas les modèles ; la reconnaissance vocale, si

Par la vision, 16/16 partout : sur des photos nettes de canettes connues, la tâche est facile pour les trois. Par la voix, 12, 13 et 14 sur 16, mais toutes les erreurs suivent une transcription fausse : quand la STT entend « toute propre » au lieu de « toute froissée », aucun modèle ne peut deviner que la canette est abîmée. Sur ces phrases abîmées, chaque modèle devine à sa façon ; l'écart de 12 à 14 ne dit rien de leur qualité. Pour départager les modèles, il faudrait des cas plus durs (photos floues, produits très proches) ; pour fiabiliser la voix, une meilleure STT, ou une confirmation (« J'ai compris : toute propre, c'est bien ça ? »).

### 2. L'inférence du LLM génératif est 16 à 18 fois plus lente, et instable

- **Par la vision, sur le robot** : 273 ms pour Jev, 394 ms pour Luna Decisions, 6 153 ms pour le génératif, soit 18 fois la moyenne des deux modèles de décision.
- **Par la voix** : 283 ms, 283 ms et 4 447 ms, soit près de 16 fois plus lent.

C'est le temps qui compte pour un robot : celui avant sa première réaction. Reachy commence à bouger et à parler dès que l'inférence est terminée. Avec un modèle de décision, c'est en moins d'une demi-seconde ; avec le LLM génératif, il reste figé de 4 à 8 secondes.

Et cette latence varie fortement : 2,6 s d'inférence médiane sur les photos et 2,3 s sur les phrases le 17 octobre ; le 19, 4,4 s sur les phrases transcrites et de 6,2 à 7,9 s sur le robot selon le passage, avec des pointes à 9 s. Sur les mêmes passages, les modèles de décision sont restés entre 0,27 et 0,43 s en médiane. Pour un robot, une latence prévisible compte autant qu'une latence basse.

### 3. De bout en bout, l'écart reste de 2 à 4 fois

Par la vision, sur le robot, l'interaction complète dure 2,1 à 2,2 s avec un modèle de décision et 8,0 s avec le génératif (3,7 fois plus). Par la voix, 3,4 s contre 7,6 s (2,2 fois plus).
La perception dépend du canal : 22 ms pour la caméra, négligeable ; près d'une seconde pour la reconnaissance vocale, soit plus de trois fois l'inférence d'un modèle de décision. Avec un modèle de décision, la réponse vocale représente encore 65 à 86 % du temps total : des répliques plus courtes, ou un geste seul pour les cas simples, feraient gagner plus que n'importe quel modèle plus rapide.

### 4. Jev contre le modèle de décision d'OpenAI : impossible de les départager

Même API (seul le nom du modèle change), même justesse, coûts proches. Et l'ordre de vitesse s'inverse d'un passage à l'autre :

| Passage (vision) | Jev, médiane | Luna Decisions, médiane |
|---|---|---|
| Photos, 1er passage | 475 ms | 567 ms |
| Photos, 2e passage | 343 ms | 328 ms |
| Robot, canette validée | 273 ms | 394 ms |
| Robot, mauvaise commande | 352 ms | 426 ms |
| Robot, scène vide | 359 ms | 400 ms |

Sur les photos, l'écart reste dans le bruit du réseau ; sur le robot, Jev a été plus rapide aux trois passages, de 40 à 120 ms. Pas de quoi changer l'expérience : les deux réagissent en moins d'une demi-seconde. Au 2e passage, Jev a eu des pics isolés jusqu'à environ 1 s (écart-type de 177 ms, contre 89 ms pour OpenAI) ; au 1er, c'était l'inverse. Les deux ont surtout un comportement différent :

- **La lecture littérale.** Lors d'un essai hors banc, la commande disait « Sierra Mist » et la photo montrait une **Diet** Sierra Mist. Jev a répondu « mauvais produit » (p = 0,08), les deux modèles d'OpenAI « bon produit ». En préparation de commandes, une Diet est une autre référence : Jev a raison. Ailleurs, ce serait un faux refus. Il faut écrire des consignes aussi précises que les références.
- **La confiance.** Avec la photo, Luna Decisions tranche plus nettement. Sur l'état de la Fanta, ouverte mais pas déformée, Jev descend à 0,52 de confiance, contre 0,92 pour OpenAI : le cas est vraiment ambigu, et c'est ce qui déclenche le « je ne suis pas sûr ». Sur la phrase exacte du préparateur (17 octobre), c'est l'inverse sur un cas : OpenAI hésite (0,60) devant « la canette est toute froissée ».
- **Le refus.** Sur le robot, sans article devant la caméra, Luna Decisions a refusé de répondre 5 fois sur 8 (« OpenAI refused to answer question 'etat' », erreur 502) ; Jev a répondu à chaque fois. Avec une canette devant la caméra, aucun refus. Un refus, c'est un robot qui ne réagit pas : le code doit le prévoir (ici, Reachy prend un air perplexe).
- **La boîte.** Seul Jev sait encadrer un objet dans l'image (question `box`) : voir la piste ci-dessous.

### 5. Le coût : la promesse ne tient pas sur l'image, elle tient à moitié sur le texte

TypeSafe annonce des décisions « 40 à 400 fois moins chères » que les LLM. Ce qu'on mesure :

- **Par la vision** : 83 $, 90 $ et 116 $ par million de contrôles sur le robot (97 $, 103 $ et 134 $ sur les photos plus grandes du 17 octobre). Le génératif n'est que 1,4 fois plus cher. On paie surtout l'entrée, et l'entrée, c'est d'abord l'image ; la réponse est un minuscule JSON, donc la sortie gratuite des modèles de décision change peu.
- **Par la voix** : 17 $, 33 $ et 70 $ par million de décisions. Là, Jev est vraiment moins cher : 2 fois moins que le modèle d'OpenAI, 4 fois moins que le génératif (2,8 fois le 17 octobre, à 47 $ : le coût du génératif varie avec le fournisseur choisi par OpenRouter). C'est réel, mais loin de 40 à 400. Et la transcription coûte à elle seule 32 $ par million de phrases, presque deux fois la décision de Jev : sur le pipeline complet, l'avantage se dilue (49 $, contre 65 $ et 102 $).
- **D'où vient l'écart avec l'annonce** : TypeSafe se compare aux LLM de pointe, bien plus chers (Claude Sonnet 5.5 coûte 2 $ en entrée et 10 $ en sortie par million de tokens), et sur des tâches où le LLM écrit davantage. Pour trancher un oui ou un non sur un robot, on prendrait plutôt un petit LLM rapide, et l'écart réel est alors de 1,4 à 4.

### 6. Pourquoi Jev compte deux fois plus de tokens (testé)

Pour la même photo, Jev facture environ 2 300 tokens et OpenAI environ 1 000. Pour comprendre, j'ai envoyé la même requête avec 1, 2 puis 4 questions :

| Tokens d'entrée facturés | 1 question | 2 questions | 4 questions |
|---|---|---|---|
| Jev, avec la photo | 1 888 | 1 903 | 1 929 |
| OpenAI Decisions, avec la photo | 754 | 903 | 1 201 |
| Jev, texte seul | 302 | 314 | 338 |
| OpenAI Decisions, texte seul | 181 | 330 | 628 |

Deux effets inverses :
- **Jev compte l'image près de 3 fois plus cher en tokens** (environ 1 600 tokens, contre environ 600 chez OpenAI), plus un fixe d'une centaine de tokens. TypeSafe ne publie pas comment il découpe les images.
- **Mais chez Jev, une question de plus ne coûte presque rien** (environ 12 tokens), contre environ 150 tokens par question chez OpenAI, avec ou sans image. Selon sa documentation, Jev lit l'état une fois et y pose toutes les questions.

Avec deux questions et une image, les deux effets s'annulent presque. Plus on pose de questions sur la même entrée, plus Jev devient avantageux : c'est le « fan-out » que recommande TypeSafe.

### 7. Ce que j'en retiens pour un robot

- Pour un réflexe (valider, refuser, réagir), un modèle de décision suffit, et le robot réagit en un tiers de seconde, de façon prévisible.
- La perception n'est pas gratuite partout : la caméra coûte 22 ms, la voix près d'une seconde, et une phrase sur deux mal comprise. Avant de changer de modèle, il faut fiabiliser l'oreille du robot.
- Jev et le modèle d'OpenAI se valent sur ce cas ; le choix se fera sur le comportement (lecture littérale, boîte) et sur le nombre de questions posées.
- Sur des images, il ne faut pas compter sur un gros gain de coût. Ce qui change l'expérience, c'est la latence, et, une fois la décision rapide, la longueur de ce que dit le robot.
- L'IA générative reste utile quand il faut des mots (expliquer, converser), pas pour trancher.

## Vérifié sur le vrai Reachy

### La perception : la caméra ne coûte presque rien (19 octobre 2026)

`uv run reachy-jev capture` chronomètre, sur le Reachy Mini Wireless, 20 prises de vue neuves (une image différente de la précédente, pas la dernière image déjà reçue), puis leur encodage pour l'IA. Aucune image n'est enregistrée.

| Mesure | Résultat |
|---|---|
| Résolution | 1 280 × 720 |
| Première image après connexion | 14 ms |
| Image neuve (médiane, pire cas) | 22 ms, 26 ms |
| Encodage pour l'IA (médiane) | 10 ms |

Cette mesure part du flux vidéo déjà ouvert : elle ne compte pas le délai de transport entre la caméra et le Mac (latence « glass-to-glass »), que cette méthode ne peut pas mesurer. Le détail est dans [`resultats/capture.json`](../resultats/capture.json).

### Le pipeline complet, modèle par modèle (19 octobre 2026)

`uv run reachy-jev bout-en-bout` enchaîne, pour chaque modèle et à chaque cycle, une prise de vue neuve, son encodage, l'inférence, puis la réplique de Reachy : 8 cycles par modèle, ordre des modèles tournant, appels un par un. Trois passages :

| Devant la caméra, commande | Jev 1.14 | GPT-6 Luna Decisions | GPT-6 Luna (génératif) |
|---|---|---|---|
| San Pellegrino, « 1 canette de San Pellegrino » | validé 8/8, 273 ms | validé 8/8, 394 ms | validé 8/8, 6 153 ms |
| San Pellegrino, « 1 canette de Coca-Cola » | mauvais produit 8/8, 352 ms | mauvais produit 8/8, 426 ms | mauvais produit 8/8, 7 893 ms |
| Rien, « 1 canette de Coca-Cola » | 359 ms | 400 ms, 5 refus sur 8 | 7 425 ms |

Inférence médiane ; réactions non relevées lors du passage à vide.

- **Les trois modèles ne se sont jamais trompés sur la vraie canette** : 48 réactions justes sur 48, qu'elle corresponde à la commande ou non.
- **La préparation de l'image ne dépend pas du modèle** : 6 à 14 ms de capture et 10 à 12 ms d'encodage en médiane, quel que soit le modèle (image ramenée à 1 024 × 576, JPEG d'environ 70 Ko, envoyée identique aux trois).
- **Ce qui diffère est côté serveur**, et c'est compris dans l'inférence : chaque fournisseur découpe l'image en tokens à sa façon. Pour la même image, Jev facture environ 1 985 tokens, OpenAI environ 900 (voir l'analyse 6).

Détail : [`resultats/bout_en_bout.json`](../resultats/bout_en_bout.json), [`bout_en_bout_mauvaise_commande.json`](../resultats/bout_en_bout_mauvaise_commande.json) et [`bout_en_bout_scene_vide.json`](../resultats/bout_en_bout_scene_vide.json). Quand `resultats/bout_en_bout.json` existe, `banc --redessiner` prend ses latences et ses coûts pour la vision, à la place de ceux des photos (l'exactitude reste celle des 16 contrôles).

## Piste : la boîte, pour que Reachy montre l'article du regard (non implémentée)

**L'idée.** Sur un poste de préparation, plusieurs articles sont posés devant Reachy. Le préparateur demande « Lequel je prends pour la commande 42 ? » : Reachy repère l'article dans l'image de sa caméra, **tourne la tête vers lui** et dit « Celui-ci ». C'est un « pick-to-light » sans installation, où le regard du robot remplace le voyant lumineux.

**Comment ça marcherait.**
1. Une requête à Jev, avec l'image de la caméra et deux questions : un Noul « le produit commandé est-il sur la table ? » et une Box « le produit commandé ». La Box renvoie toujours une boîte, même si l'objet est absent : d'où le Noul.
2. Le centre de la boîte donne le point à regarder, que `look_at_image(u, v)` du SDK transforme en mouvement de tête.
3. S'il est absent, Reachy secoue la tête : « Je ne le vois pas. »

**Ce qu'il faudrait mesurer.** La précision des boîtes, en comparant avec des boîtes tracées à la main ; la latence, qui ajoute un mouvement de tête ; et la comparaison avec un LLM génératif à qui l'on demande des coordonnées, puisque le modèle d'OpenAI ne sait pas faire de boîte.

**Les points de vigilance.**
- Dans mes essais, la confiance de la boîte était parfois basse (0,25) alors que la boîte était juste : il ne faut pas s'y fier seule.
- Le SDK prévient que `look_at_image` n'est pas encore exact avec certaines caméras : à calibrer sur le robot.
- Toute requête avec image passe chez Jev dans un mode annoncé plus lent.

## Limites

- 16 contrôles par mode, sur des photos de Wikimedia Commons : l'exactitude n'est mesurée que sur ces photos. Sur le robot, une seule canette a été montrée (48 réactions justes sur 48) : c'est une vérification, pas une mesure d'exactitude.
- La latence est mesurée depuis un Mac en Wi-Fi, via OpenRouter : elle inclut le réseau et le routage, comme pour un vrai robot. Deux passages sur les photos et trois sur le robot, le même jour : pas assez pour classer finement les deux modèles de décision.
- La voix du préparateur est synthétique (macOS « Jacques ») et enregistrée au calme : une vraie voix, dans un entrepôt bruyant, serait sans doute plus mal transcrite. Une seule STT a été testée ; une autre donnerait d'autres chiffres.
- La durée de la phrase dépend de la voix (ici macOS « Thomas ») et du verdict (de 1,8 à 2,5 s).
- GPT-6 Luna génératif tourne dans son réglage le plus rapide (raisonnement « minimal ») ; un autre modèle génératif donnerait d'autres chiffres.
- L'API Decisions est en alpha, et l'envoi d'images est une fonction expérimentale chez TypeSafe : formats et résultats peuvent changer.

## Sources

- [Jev sur OpenRouter](https://openrouter.ai/docs/guides/community/jev) et [les images dans l'API Decisions](https://openrouter.ai/docs/guides/community/multimodal-decisions)
- [Documentation TypeSafe](https://docs.typesafe.ai) : questions, confiance, modèles
- [GPT-6 Luna Decisions](https://openrouter.ai/openai/gpt-6-luna-decisions), [GPT-6 Luna](https://openrouter.ai/openai/gpt-6-luna) et [Jev 1.14](https://openrouter.ai/typesafe/jev-1.14) : prix et fournisseurs
