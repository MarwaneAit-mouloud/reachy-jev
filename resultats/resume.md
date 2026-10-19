# Résultats

Généré par `uv run reachy-jev banc`. Modèles : `typesafe/jev-1.14`, `openai/gpt-6-luna-decisions`, `openai/gpt-6-luna`.

## Par la vision (photo de l'article)

Perception : capture 11 ms + encodage 11 ms.

| Modèle | Exactitude | Bon produit | État | Doutes | Inférence (médiane) | Pire inférence | Perception | Réponse vocale | Latence de bout en bout | Coût par requête | Coût par million | Tokens d'entrée |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.14 | **16/16** | 16/16 | 16/16 | 0 | **273 ms** | 441 ms | 22 ms | 1,8 s | **2,1 s** | 0,000083 $ | **83 $** | 1 985 |
| GPT-6 Luna Decisions | **16/16** | 16/16 | 16/16 | 0 | **394 ms** | 544 ms | 22 ms | 1,8 s | **2,2 s** | 0,000090 $ | **90 $** | 901 |
| GPT-6 Luna (génératif) | **16/16** | 16/16 | 16/16 | 0 | **6 153 ms** | 8 572 ms | 19 ms | 1,8 s | **8,0 s** | 0,000116 $ | **116 $** | 882 |

## Par la voix (le préparateur décrit l'article)

Perception : reconnaissance vocale (STT) 0,97 s.

| Modèle | Exactitude | Bon produit | État | Doutes | Inférence (médiane) | Pire inférence | Perception | Réponse vocale | Latence de bout en bout | Coût par requête | Coût par million | Tokens d'entrée |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.14 | **12/16** | 15/16 | 10/16 | 5 | **283 ms** | 488 ms | 966 ms | 2,2 s | **3,4 s** | 0,000017 $ | **17 $** | 405 |
| GPT-6 Luna Decisions | **13/16** | 15/16 | 11/16 | 3 | **283 ms** | 1 079 ms | 966 ms | 2,2 s | **3,4 s** | 0,000033 $ | **33 $** | 334 |
| GPT-6 Luna (génératif) | **14/16** | 15/16 | 12/16 | 0 | **4 447 ms** | 8 774 ms | 966 ms | 2,2 s | **7,6 s** | 0,000070 $ | **70 $** | 209 |

Exactitude : toutes les erreurs (9) portent sur une phrase mal transcrite par la STT.

*Par la vision (photo de l'article) : Source : Expérimentations avec OpenRouter (pipeline mesuré sur le Reachy Mini face à une vraie canette, médiane de 8 cycles par modèle ; exactitude sur 16 contrôles). TTS : voix macOS « Thomas ».*

*Par la voix (le préparateur décrit l'article) : Source : Expérimentations avec OpenRouter (voix du préparateur synthétisée puis transcrite par gemini-2.5-flash-lite (STT) ; appels séquentiels). TTS : voix macOS « Thomas ».*
