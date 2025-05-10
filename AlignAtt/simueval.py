class SimuEval:
    def __init__(self):
        self.latency_scorer = []
        self.quality_scorer = []

        self.chars__latency = 0
        self.words__latency = 0
        self.tokens_latency = 0

        self.cs = 0

    def update(self, partial_input_text, full_input_text, predicted_text,
               gold_text, bpe_tokenizer):
        # shortened names for partial_input_text, full_input_text, predicted_text, gold_text, bpe_tokenizer
        pit, fit, pt, gt, t = partial_input_text, full_input_text, predicted_text, gold_text, bpe_tokenizer
        chars_delay = self.calc_latency_ratio(pit, fit, pt, gt)

        words_pit, words_fit, words_pt, words_gt = pit.split(" "), fit.split(" "), pt.split(" "), gt.split(" ")
        words_delay = self.calc_latency_ratio(words_pit, words_fit, words_pt, words_gt)

        tokens_delay = self.calc_latency_ratio(words_pit, words_fit, words_pt, words_gt)

        new_delay_tokens = (frac_tokens * len(
            bpe_tokenizer.tokenize(partial_input_text))) - len(
            bpe_tokenizer.tokenize(predicted_text)) / len(tokens_en)
        new_bleu = bleu.compute(input_text, gold_text)

    # return the latency ratio for given chars, words or tokens
    def calc_latency_ratio(self, partial, full, pred, gold):
        return (len(gold) / len(full) * len(partial) - len(pred)) / len(gold)

    # figure out only what g is
    def average_lagging(self, prefixes, source, target, stable_len):
        al_sum = 0
        r = len(target) / len(source)

        for i in range(1, stable_len):
            al_sum += g - (i - 1) / r

        return (1 / stable_len) * al_sum

    def eval(self):
        return ""
        return {
            "delay_chars": self.delay_chars,
            "delay_words": self.delay_words,
            "delay_tokens": self.delay_tokens,
        }
