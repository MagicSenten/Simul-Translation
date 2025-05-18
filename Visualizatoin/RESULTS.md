# Results of the experiments

## AlignAtt Baseline

| Frame Size \ Layers | 1 | 2 | 3 |
|:---:|:---:|:---:|:---:|
| 1 | BLEU: 11.53<br>WER: 0.78<br>AL: 8.90 | BLEU: 11.19<br>WER: 0.78<br>AL: 8.70 | BLEU: 13.75<br>WER: 0.79<br>AL: 8.80 |
| 2 | BLEU: 11.28<br>WER: 0.78<br>AL: 8.91 | BLEU: 11.09<br>WER: 0.78<br>AL: 8.72 | BLEU: 14.30<br>WER: 0.78<br>AL: 8.89 |
| 3 | BLEU: 11.10<br>WER: 0.78<br>AL: 8.94 | BLEU: 10.92<br>WER: 0.78<br>AL: 8.73 | BLEU: 14.53<br>WER: 0.77<br>AL: 9.06 |

## AlignAtt Finetuned

| Frame Size \ Layers | 1 | 2 | 3 |
|:---:|:---:|:---:|:---:|
| 1 | BLEU: 8.34<br>WER: 0.78<br>AL: 9.06 | BLEU: 8.92<br>WER: 0.77<br>AL: 9.06 | BLEU: 15.19<br>WER: 0.74<br>AL: 9.37 |
| 2 | BLEU: 8.24<br>WER: 0.78<br>AL: 9.05 | BLEU: 8.80<br>WER: 0.77<br>AL: 9.08 | BLEU: 15.27<br>WER: 0.74<br>AL: 9.39 |
| 3 | BLEU: 8.07<br>WER: 0.78<br>AL: 9.08 | BLEU: 8.87<br>WER: 0.77<br>AL: 9.13 | BLEU: 15.15<br>WER: 0.74<br>AL: 9.36 |

## Local Agreement MNT Baseline

| #Beams \ Wait At Begin | 1 | 2 | 3 |
|:---:|:---:|:---:|:---:|
| 1 | BLEU: 11.43<br>WER: 0.87<br>AL: 7.75 | BLEU: 13.09<br>WER: 0.83<br>AL: 7.82 | BLEU: 12.40<br>WER: 0.83<br>AL: 8.20 |
| 2 | BLEU: 10.44<br>WER: 0.89<br>AL: 7.88 | BLEU: 11.09<br>WER: 0.86<br>AL: 8.08 | BLEU: 11.29<br>WER: 0.82<br>AL: 8.40 |
| 3 | BLEU: 9.99<br>WER: 0.90<br>AL: 7.95 | BLEU: 11.18<br>WER: 0.87<br>AL: 8.14 | BLEU: 11.64<br>WER: 0.83<br>AL: 8.41 |

## Local Agreement Finetuned

| #Beams \ Wait At Begin | 1 | 2 | 3 |
|:---:|:---:|:---:|:---:|
| 1 | BLEU: 15.14<br>WER: 0.77<br>AL: 9.40 | BLEU: 15.40<br>WER: 0.76<br>AL: 9.44 | BLEU: 16.13<br>WER: 0.75<br>AL: 9.60 |
| 2 | BLEU: 15.39<br>WER: 0.77<br>AL: 9.37 | BLEU: 15.67<br>WER: 0.76<br>AL: 9.50 | BLEU: 15.89<br>WER: 0.75<br>AL: 9.53 |
| 3 | BLEU: 15.34<br>WER: 0.77<br>AL: 9.34 | BLEU: 15.15<br>WER: 0.76<br>AL: 9.47 | BLEU: 16.07<br>WER: 0.74<br>AL: 9.53 |

## Local Agreement LLM

| #Beams \ Wait At Begin | 1 | 2 | 3 |
|:---:|:---:|:---:|:---:|
| 1 | BLEU: 11.33<br>WER: 0.79<br>AL: 9.33 | BLEU: 11.62<br>WER: 0.79<br>AL: 9.73 | BLEU: 11.02<br>WER: 0.79<br>AL: 10.25 |
| 2 | BLEU: 11.34<br>WER: 0.80<br>AL: 9.29 | BLEU: 11.78<br>WER: 0.79<br>AL: 9.75 | BLEU: 11.44<br>WER: 0.80<br>AL: 10.23 |
| 3 | BLEU: 11.14<br>WER: 0.80<br>AL: 9.33 | BLEU: 11.58<br>WER: 0.79<br>AL: 9.73 | BLEU: 11.55<br>WER: 0.79<br>AL: 10.22 |

## Finetuned

| #Beams \ Wait At Begin | 1 | 2 | 3 |
|:---:|:---:|:---:|:---:|
| 1 | BLEU: 17.01<br>WER: 0.80<br>AL: 8.71 | BLEU: 17.21<br>WER: 0.78<br>AL: 8.88 | BLEU: 17.04<br>WER: 0.77<br>AL: 9.13 |
| 2 | BLEU: 16.79<br>WER: 0.78<br>AL: 8.72 | BLEU: 16.82<br>WER: 0.78<br>AL: 8.92 | BLEU: 17.62<br>WER: 0.76<br>AL: 9.06 |
| 3 | BLEU: 16.84<br>WER: 0.79<br>AL: 8.70 | BLEU: 16.90<br>WER: 0.78<br>AL: 8.93 | BLEU: 17.53<br>WER: 0.76<br>AL: 9.14 |

## MNT Backbone

| #Beams \ Wait At Begin | 1 | 2 | 3 |
|:---:|:---:|:---:|:---:|
| 1 | BLEU: 11.43<br>WER: 0.87<br>AL: 7.75 | BLEU: 13.09<br>WER: 0.83<br>AL: 7.82 | BLEU: 12.40<br>WER: 0.83<br>AL: 8.20 |
| 2 | BLEU: 10.44<br>WER: 0.89<br>AL: 7.88 | BLEU: 11.09<br>WER: 0.86<br>AL: 8.08 | BLEU: 11.29<br>WER: 0.82<br>AL: 8.40 |
| 3 | BLEU: 9.99<br>WER: 0.90<br>AL: 7.95 | BLEU: 11.18<br>WER: 0.87<br>AL: 8.14 | BLEU: 11.64<br>WER: 0.83<br>AL: 8.41 |
