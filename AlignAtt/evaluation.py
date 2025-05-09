class Metric:
    def __init__(self):
        self.delay_chars = 0
        self.delay_words = 0
        self.delay_tokens = 0
        self.cs = 0

    def update(self, input_text, predicted_text, gold_text, bpe_tokenizer):
        new_delay_chars = (frac_chars * len(gold_text)) - len(predicted_text) / lhten
        new_delay_words = (frac_words * len(gold_text.split(" "))) - len(predicted_text.split(" ")) / l_tokens
        new_delay_tokens = (frac_tokens * len(bpe_tokenizer.tokenize(gold_text[0]))) - len(bpe_tokenizer.tokenize(predicted_text)) / len(tokens_en)
        return new_delay_chars, new_delay_words, new_delay_tokens\

    def eval(self):
        return {
            "delay_chars": self.delay_chars,
            "delay_words": self.delay_words,
            "delay_tokens": self.delay_tokens,
        }