
## AlignAtt Baseline

| Frame Size |1 (Layers)|2 (Layers)|3 (Layers)|
|----|----|----|----|
| 1 | BLEU: 11.53, WER: 0.78, AL: 8.90 | BLEU: 11.19, WER: 0.78, AL: **8.70** | BLEU: 13.75, WER: 0.79, AL: 8.80 |
| 2 | BLEU: 11.28, WER: 0.78, AL: 8.91 | BLEU: 11.09, WER: 0.78, AL: 8.72 | BLEU: 14.30, WER: 0.78, AL: 8.89 |
| 3 | BLEU: 11.10, WER: 0.78, AL: 8.94 | BLEU: 10.92, WER: 0.78, AL: 8.73 | BLEU: **14.53**, WER: **0.77**, AL: 9.06 |

## AlignAtt Finetuned

| Frame Size |1 (Layers)|2 (Layers)|3 (Layers)|
|----|----|----|----|
| 1 | BLEU: 8.34, WER: 0.78, AL: 9.06 | BLEU: 8.92, WER: 0.77, AL: 9.06 | BLEU: 15.19, WER: 0.74, AL: 9.37 |
| 2 | BLEU: 8.24, WER: 0.78, AL: **9.05** | BLEU: 8.80, WER: 0.77, AL: 9.08 | BLEU: 15.27, WER: 0.74, AL: 9.39 |
| 3 | BLEU: 8.07, WER: 0.78, AL: 9.08 | BLEU: 8.87, WER: 0.77, AL: 9.13 | BLEU: 15.15, WER: 0.74, AL: 9.36 |

## Local Agreement MNT Baseline

| #Beams |1 (Wait At Begin)|2 (Wait At Begin)|3 (Wait At Begin)|
|----|----|----|----|
| 1 | BLEU: 11.43, WER: 0.87, AL: **7.75** | BLEU: 13.09, WER: 0.83, AL: 7.82 | BLEU: 12.40, WER: 0.83, AL: 8.20 |
| 2 | BLEU: 10.44, WER: 0.89, AL: 7.88 | BLEU: 11.09, WER: 0.86, AL: 8.08 | BLEU: 11.29, WER: 0.82, AL: 8.40 |
| 3 | BLEU: 9.99, WER: 0.90, AL: 7.95 | BLEU: 11.18, WER: 0.87, AL: 8.14 | BLEU: 11.64, WER: 0.83, AL: 8.41 |

## Local Agreement Finetuned

| #Beams |1 (Wait At Begin)|2 (Wait At Begin)|3 (Wait At Begin)|
|----|----|----|----|
| 1 | BLEU: 15.14, WER: 0.77, AL: 9.40 | BLEU: 15.40, WER: 0.76, AL: 9.44 | BLEU: **16.13**, WER: 0.75, AL: 9.60 |
| 2 | BLEU: 15.39, WER: 0.77, AL: 9.37 | BLEU: 15.67, WER: 0.76, AL: 9.50 | BLEU: 15.89, WER: 0.75, AL: 9.53 |
| 3 | BLEU: 15.34, WER: 0.77, AL: **9.34** | BLEU: 15.15, WER: 0.76, AL: 9.47 | BLEU: 16.07, WER: **0.74**, AL: 9.53 |

## Local Agreement LLM

| #Beams |1 (Wait At Begin)|2 (Wait At Begin)|3 (Wait At Begin)|
|----|----|----|----|
| 1 | BLEU: 11.33, WER: 0.79, AL: 9.33 | BLEU: 11.62, WER: **0.79**, AL: 9.73 | BLEU: 11.02, WER: 0.79, AL: 10.25 |
| 2 | BLEU: 11.34, WER: 0.80, AL: **9.29** | BLEU: **11.78**, WER: 0.79, AL: 9.75 | BLEU: 11.44, WER: 0.80, AL: 10.23 |
| 3 | BLEU: 11.14, WER: 0.80, AL: 9.33 | BLEU: 11.58, WER: 0.79, AL: 9.73 | BLEU: 11.55, WER: 0.79, AL: 10.22 |

## Finetuned

| #Beams |1 (Wait At Begin)|2 (Wait At Begin)|3 (Wait At Begin)|
|----|----|----|----|
| 1 | BLEU: 17.01, WER: 0.80, AL: 8.71 | BLEU: 17.21, WER: 0.78, AL: 8.88 | BLEU: 17.04, WER: 0.77, AL: 9.13 |
| 2 | BLEU: 16.79, WER: 0.78, AL: 8.72 | BLEU: 16.82, WER: 0.78, AL: 8.92 | BLEU: **17.62**, WER: **0.76**, AL: 9.06 |
| 3 | BLEU: 16.84, WER: 0.79, AL: **8.70** | BLEU: 16.90, WER: 0.78, AL: 8.93 | BLEU: 17.53, WER: 0.76, AL: 9.14 |

## MNT Backbone

| #Beams |1 (Wait At Begin)|2 (Wait At Begin)|3 (Wait At Begin)|
|----|----|----|----|
| 1 | BLEU: 11.43, WER: 0.87, AL: **7.75** | BLEU: **13.09**, WER: 0.83, AL: 7.82 | BLEU: 12.40, WER: 0.83, AL: 8.20 |
| 2 | BLEU: 10.44, WER: 0.89, AL: 7.88 | BLEU: 11.09, WER: 0.86, AL: 8.08 | BLEU: 11.29, WER: **0.82**, AL: 8.40 |
| 3 | BLEU: 9.99, WER: 0.90, AL: 7.95 | BLEU: 11.18, WER: 0.87, AL: 8.14 | BLEU: 11.64, WER: 0.83, AL: 8.41 |
