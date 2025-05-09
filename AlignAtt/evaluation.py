class SimuEval:
    def __init__(self):
        self.delay_chars = 0
        self.delay_words = 0
        self.delay_tokens = 0
        self.cs = 0

    def update(self, partial_input_text, full_input_text, predicted_text, gold_text, bpe_tokenizer):
        return
        new_delay_chars = ((len(gold_text) / len(full_input_text)) * len(partial_input_text) - len(predicted_text)) / len(gold_text)
        new_delay_words = (frac_words * len(gold_text.split(" "))) - len(predicted_text.split(" ")) / l_tokens
        new_delay_tokens = (frac_tokens * len(bpe_tokenizer.tokenize(partial_input_text))) - len(bpe_tokenizer.tokenize(predicted_text)) / len(tokens_en)
        new_bleu = bleu.compute(input_text, gold_text)

    def eval(self):
        return ""
        return {
            "delay_chars": self.delay_chars,
            "delay_words": self.delay_words,
            "delay_tokens": self.delay_tokens,
        }