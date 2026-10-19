# Jev en une page (état au 30 septembre 2026, mis à jour le 17 octobre)

Ce que la documentation publique dit de Jev, pour ne rien affirmer de faux dans le post.

## Nouveau depuis le 14 octobre 2026 : Jev 1.14

- `typesafe/jev-1.14` sur OpenRouter, même prix (0,042 $ par million de tokens d'entrée, sortie gratuite), contexte de 262 144 tokens.
- **Images** : Jev lit des images JPEG ou PNG (256 à 1 568 px par côté) placées dans le `state`. Fonction marquée « early » ; toute requête
  avec image passe dans le mode long contexte, annoncé plus lent et moins précis.
- **Box** : une quatrième primitive, réservée à Jev, qui renvoie la boîte d'un objet dans l'image (`top`, `bottom`, `left`, `right`, en
  fractions) et une confiance. Pas de réponse « absent » : on la double d'un Noul.
- **Concurrence** : au 16 octobre, 22 modèles de décision de 13 éditeurs sur la même API Decisions, dont `openai/gpt-6-luna-decisions`
  (0,10 $ par million de tokens d'entrée, texte et images). C'est lui que compare ce projet.
- **SDK et images via OpenRouter** : le SDK `typesafe_sdk` envoie les images en multipart, qu'OpenRouter refuse (400) ; et sans la
  référence `{photo}` dans le texte, `attach_assets` n'envoie rien, sans erreur. Ce projet appelle donc l'API Decisions en HTTP.

## Ce que c'est

- **Jev** est le premier « **System One** model » de **TypeSafe AI** (San Francisco, fondée en 2024, dirigée par Diogo Almeida, ex-OpenAI).
  Accès limité annoncé le 15 septembre 2026 avec une levée de 40 M$ ; disponible sur OpenRouter depuis fin septembre, sans liste d'attente.
- Ce n'est **pas un LLM** : il ne génère pas de texte. Il reçoit un **état** (texte, objet JSON ou tableau) et des **questions typées**, et renvoie
  des réponses typées avec des probabilités. Il ne donne ni explication ni raisonnement.
- Poids et architecture **non publiés** ; entraîné, selon TypeSafe, sur des données synthétiques.

## Les trois primitives

| Primitive | Question | Réponse |
|---|---|---|
| `choice` | Laquelle de ces options ? (jusqu'à 255) | le choix, une probabilité par option, une **confiance** |
| `noul` | Cette condition est-elle vraie ? | la probabilité du oui (pas de champ confiance) |
| `score` | Où sur cette échelle ordonnée ? (jusqu'à 10 niveaux) | la position pondérée, une probabilité par niveau, une confiance |

Toutes les questions d'une requête sont évaluées **en parallèle et en isolation** : on peut en poser beaucoup à la fois (« fan-out »).

## L'API utilisée ici

`POST https://openrouter.ai/api/alpha/decisions` (clé OpenRouter), modèle `typesafe/jev-1.14` (version épinglée) :

```json
{"model": "typesafe/jev-1.14",
 "state": {"object": "aluminium soda can", "condition": "empty"},
 "questions": {"bac": {"type": "choice", "instructions": "Which bin?", "criteria": {"jaune": "…", "verre": "…"}}}}
```

Réponse : `{"answers": {"bac": {"type": "choice", "choice": "jaune", "confidence": 0.93, "probabilities": {…}}}, "usage": {"input_tokens": …, "cost": …}}`.
Contexte : 262 144 tokens (1.14). Prix : 0,042 $ par million de tokens d'entrée, **sortie gratuite** ; la réponse indique le coût réel (`usage.cost`).
L'endpoint est en **alpha** : le format peut bouger. Le chemin `typesafe/jev-router` (chat, accepte les images) est un autre produit : un routeur de modèles.

## Ce que TypeSafe annonce (à présenter comme leurs chiffres)

- 70 à 500 ms de bout en bout ; « 40 à 200 fois plus rapide » et « 40 à 400 fois moins cher » que les LLM de pointe sur des tâches comparables.
- TypeSafe reconnaît que ces mesures viennent de flux de travail construits en interne et que les gains réels seront probablement plus faibles.

## Ce qu'en disent d'autres

- Un test indépendant publié sur *Towards Data Science* (Banking77, 77 intentions) : Jev 81,1 % contre 76,4 % pour un LLM open source ; latence
  équivalente (~245 ms) ; 1,72 % d'étiquettes invalides côté LLM ; **surconfiance** de Jev dans les confiances intermédiaires ; une cascade Jev → LLM
  a fait **baisser** la précision. Leçon reprise ici : on teste sur SES données, avec calibration, et on évalue la cascade avant de s'y fier.
- Des développeurs cités par la presse : 5 à 18 fois plus rapide qu'un modèle OpenAI sur une classification de sécurité ; ou 10 à 20 fois **plus cher**
  qu'un Gemini léger. D'où la présence, dans ce benchmark, des LLM les moins chers (Flash-Lite, Luna) et pas seulement des modèles de pointe.
- Critique d'Armin Ronacher : le modèle « délègue le problème d'hallucination à l'utilisateur » (c'est à toi de choisir le seuil de confiance).

## Faiblesses documentées par TypeSafe (« Jev 1.13 jaggedness »)

Lecture littérale des consignes · **chiffres et calcul** · **comparaison de dates** · indirection (double négation) · contexte inutile qui dégrade ·
**contenu adversarial** (une instruction glissée dans l'état peut faire bouger la réponse) · consignes contradictoires · invariants de bon sens non garantis ·
génération de texte. Recommandation de TypeSafe : garder l'arithmétique dans le code, extraire avec Jev puis calculer. Ce projet applique
la même idée au verdict : Jev répond à deux questions simples, le code tranche.

## Sources

- [Jev sur OpenRouter (hub, tutoriel, référence de l'API)](https://openrouter.ai/docs/guides/community/jev)
- [Documentation TypeSafe (concepts, primitives, confiance)](https://docs.typesafe.ai) et [la page « jaggedness »](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- [Jev (AI model) — Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model))
- [TechCrunch, 18 septembre 2026](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/)
- [Jev vs. LLMs — Towards Data Science](https://towardsdatascience.com/jev-vs-llms-when-ai-moves-from-generation-to-decision-making/)
