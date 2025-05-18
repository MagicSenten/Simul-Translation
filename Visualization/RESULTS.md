
## AlignAtt NMT Baseline

| Frame Size |4. (Layer)|3. (Layer)|2. (Layer)|
|----|----|----|----|
| 1 | BLEU: 11.53, AL: 8.90, WER: 0.78 | BLEU: 11.19, AL: **8.70**, WER: 0.78 | BLEU: 13.75, AL: 8.80, WER: 0.79 |
| 2 | BLEU: 11.28, AL: 8.91, WER: 0.78 | BLEU: 11.09, AL: 8.72, WER: 0.78 | BLEU: 14.30, AL: 8.89, WER: 0.78 |
| 3 | BLEU: 11.10, AL: 8.94, WER: 0.78 | BLEU: 10.92, AL: 8.73, WER: 0.78 | BLEU: **14.53**, AL: 9.06, WER: **0.77** |

## AlignAtt NMT Finetuned

| Frame Size |4. (Layer)|3. (Layer)|2. (Layer)|
|----|----|----|----|
| 1 | BLEU: 8.34, AL: 9.06, WER: 0.78 | BLEU: 8.92, AL: 9.06, WER: 0.77 | BLEU: 15.19, AL: 9.37, WER: **0.74** |
| 2 | BLEU: 8.24, AL: **9.05**, WER: 0.78 | BLEU: 8.80, AL: 9.08, WER: 0.77 | BLEU: **15.27**, AL: 9.39, WER: **0.74** |
| 3 | BLEU: 8.07, AL: 9.08, WER: 0.78 | BLEU: 8.87, AL: 9.13, WER: 0.77 | BLEU: 15.15, AL: 9.36, WER: **0.74** |

## Local Agreement NMT Baseline

| #Beams |1 (Wait At Begin)|2 (Wait At Begin)|3 (Wait At Begin)|
|----|----|----|----|
| 1 | BLEU: 11.43, AL: **7.75**, WER: 0.87 | BLEU: **13.09**, AL: 7.82, WER: 0.83 | BLEU: 12.40, AL: 8.20, WER: 0.83 |
| 2 | BLEU: 10.44, AL: 7.88, WER: 0.89 | BLEU: 11.09, AL: 8.08, WER: 0.86 | BLEU: 11.29, AL: 8.40, WER: **0.82** |
| 3 | BLEU: 9.99, AL: 7.95, WER: 0.90 | BLEU: 11.18, AL: 8.14, WER: 0.87 | BLEU: 11.64, AL: 8.41, WER: 0.83 |

## Local Agreement NMT Finetuned

| #Beams |1 (Wait At Begin)|2 (Wait At Begin)|3 (Wait At Begin)|
|----|----|----|----|
| 1 | BLEU: 15.14, AL: 9.40, WER: 0.77 | BLEU: 15.40, AL: 9.44, WER: 0.76 | BLEU: **16.13**, AL: 9.60, WER: 0.75 |
| 2 | BLEU: 15.39, AL: 9.37, WER: 0.77 | BLEU: 15.67, AL: 9.50, WER: 0.76 | BLEU: 15.89, AL: 9.53, WER: 0.75 |
| 3 | BLEU: 15.34, AL: **9.34**, WER: 0.77 | BLEU: 15.15, AL: 9.47, WER: 0.76 | BLEU: 16.07, AL: 9.53, WER: **0.74** |

## Local Agreement LLM

| #Beams |1 (Wait At Begin)|2 (Wait At Begin)|3 (Wait At Begin)|
|----|----|----|----|
| 1 | BLEU: 11.33, AL: 9.33, WER: **0.79** | BLEU: 11.62, AL: 9.73, WER: **0.79** | BLEU: 11.02, AL: 10.25, WER: **0.79** |
| 2 | BLEU: 11.34, AL: **9.29**, WER: 0.80 | BLEU: **11.78**, AL: 9.75, WER: **0.79** | BLEU: 11.44, AL: 10.23, WER: 0.80 |
| 3 | BLEU: 11.14, AL: 9.33, WER: 0.80 | BLEU: 11.58, AL: 9.73, WER: **0.79** | BLEU: 11.55, AL: 10.22, WER: **0.79** |

## NMT Finetuned

| #Beams |1 (Wait At Begin)|2 (Wait At Begin)|3 (Wait At Begin)|
|----|----|----|----|
| 1 | BLEU: 17.01, AL: 8.71, WER: 0.80 | BLEU: 17.21, AL: 8.88, WER: 0.78 | BLEU: 17.04, AL: 9.13, WER: 0.77 |
| 2 | BLEU: 16.79, AL: 8.72, WER: 0.78 | BLEU: 16.82, AL: 8.92, WER: 0.78 | BLEU: **17.62**, AL: 9.06, WER: **0.76** |
| 3 | BLEU: 16.84, AL: **8.70**, WER: 0.79 | BLEU: 16.90, AL: 8.93, WER: 0.78 | BLEU: 17.53, AL: 9.14, WER: **0.76** |

## NMT Baseline

| #Beams |1 (Wait At Begin)|2 (Wait At Begin)|3 (Wait At Begin)|
|----|----|----|----|
| 1 | BLEU: 11.43, AL: **7.75**, WER: 0.87 | BLEU: **13.09**, AL: 7.82, WER: 0.83 | BLEU: 12.40, AL: 8.20, WER: 0.83 |
| 2 | BLEU: 10.44, AL: 7.88, WER: 0.89 | BLEU: 11.09, AL: 8.08, WER: 0.86 | BLEU: 11.29, AL: 8.40, WER: **0.82** |
| 3 | BLEU: 9.99, AL: 7.95, WER: 0.90 | BLEU: 11.18, AL: 8.14, WER: 0.87 | BLEU: 11.64, AL: 8.41, WER: 0.83 |
