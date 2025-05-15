class SimuEval:
    def __init__(self):
        self.latency_scorer = []
        self.quality_scorer = []

        self.chars__latency = [0, 0]
        self.words__latency = [0, 0]
        self.tokens_latency = [0, 0]

        self.cs = 0


    def update(self, inputs, outputs, gold_text, bpe_tokenizer):
        # shortened names for partial_input_text, full_input_text, predicted_text, gold_text, bpe_tokenizer
        for x,y in zip(inputs, outputs):
            pit, fit, pt, gt, t = x, inputs[-1], y, gold_text, bpe_tokenizer
            chars_delay = self.calc_latency_ratio(pit, fit, pt, gt)
            self.chars__latency[0] += chars_delay[0]
            self.chars__latency[1] += chars_delay[1]

            words_pit, words_fit, words_pt, words_gt = pit.split(" "), fit.split(" "), pt.split(" "), gt.split(" ")
            words_delay = self.calc_latency_ratio(words_pit, words_fit, words_pt, words_gt)
            self.words__latency[0] += words_delay[0]
            self.words__latency[1] += words_delay[1]

            tokens_pit, tokens_fit, tokens_pt, tokens_gt = t.tokenize(pit), t.tokenize(fit), t.tokenize(pt), t.tokenize(gt)
            tokens_delay = self.calc_latency_ratio(tokens_pit, tokens_fit, tokens_pt, tokens_gt)
            self.tokens_latency[0] += tokens_delay[0]
            self.tokens_latency[1] += tokens_delay[1]
            self.cs += 1

    # return the latency ratio for given chars, words or tokens
    def calc_latency_ratio(self, partial, full,  pred, gold):
        absolute = max(len(gold) / len(full) * len(partial) - len(pred), 0)
        return absolute / len(gold), absolute

    def eval(self):
        return {
            "delay_chars": self.chars__latency[0] / self.cs,
            "delay_words": self.words__latency[0] / self.cs,
            "delay_tokens": self.tokens_latency[0] / self.cs,
            "delay_chars_absolute": self.chars__latency[1] / self.cs,
            "delay_words_absolute": self.words__latency[1] / self.cs,
            "delay_tokens_absolute": self.tokens_latency[1] / self.cs,
        }