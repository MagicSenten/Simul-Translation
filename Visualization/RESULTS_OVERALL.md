## Results on all data.
 | experiment name | bleu | word error rate | average lagging|
 | ---- | ---- | ---- | ----|
| LLM local agreement | 11.33 | 0.79 | 9.33 | 
| baseline alignatt | 14.30 | 0.78 | 8.89 | 
| baseline local agreement | 16.53 | 0.78 | 9.11 | 
| baseline wait beginning | 13.09 | 0.83 | **7.82** | 
| finetuned alignatt | 16.43 | **0.73** | 9.36 | 
| finetuned local agreement | 16.07 | 0.74 | 9.53 | 
| finetuned wait beginning | **17.62** | 0.76 | 9.06 | 
## Results on all sentences shorter than 100 cahracters.
 | experiment name | bleu | word error rate | average lagging|
 | ---- | ---- | ---- | ----|
| LLM local agreement | 11.37 | 0.79 | 14.55 | 
| baseline alignatt | 14.62 | 0.78 | 13.65 | 
| baseline local agreement | 17.27 | 0.77 | 13.72 | 
| baseline wait beginning | 12.16 | 0.83 | **11.81** | 
| finetuned alignatt | 16.49 | **0.74** | 14.26 | 
| finetuned local agreement | 15.61 | 0.75 | 14.29 | 
| finetuned wait beginning | **17.84** | 0.75 | 13.56 | 
## Results on all sentences longer than 100 cahracters.
 | experiment name | bleu | word error rate | average lagging|
 | ---- | ---- | ---- | ----|
| LLM local agreement | 11.37 | 0.79 | 7.16 | 
| baseline alignatt | 13.81 | 0.78 | 6.94 | 
| baseline local agreement | 15.37 | 0.79 | 7.14 | 
| baseline wait beginning | 13.03 | 0.84 | **6.16** | 
| finetuned alignatt | 16.06 | **0.73** | 7.30 | 
| finetuned local agreement | 16.95 | 0.74 | 7.57 | 
| finetuned wait beginning | **16.95** | 0.77 | 7.13 | 
