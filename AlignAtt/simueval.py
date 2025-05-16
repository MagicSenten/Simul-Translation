from typing import List, Union
import sacrebleu


def compute(
        delays: List[Union[float, int]],
        source_length: Union[float, int],
        target_length: Union[float, int],
):
    """
    Function to compute latency on one sentence (instance).

    Args:
        delays (List[Union[float, int]]): Sequence of delays.
        source_length (Union[float, int]): Length of source sequence.
        target_length (Union[float, int]): Length of target sequence.

    Returns:
        float: the latency score on one sentence.
    """

    if delays[0] > source_length:
        return delays[0]

    AL = 0
    gamma = target_length / source_length
    tau = 0
    for t_miuns_1, d in enumerate(delays):
        if d <= source_length:
            AL += d - t_miuns_1 / gamma
            tau = t_miuns_1 + 1

            if d == source_length:
                break
    AL /= tau
    return AL


class SimuEval:
    def __init__(self):
        self.counter = 0
        self.delays = []

        self.predictions = []
        self.golden_trans = []

        self._AL = []
        self.bleu = 0

        # self.latency_scorer = []
        # self.quality_scorer = []

        self.words__latency = 0

    def update(self, inputs, pred_outputs, gold_text, tokenizer):
        delays = []

        # save the data for the sacreBLEU evaluation
        self.predictions.append(pred_outputs[-1])
        self.golden_trans.append(gold_text)

        # # prints to be deleted
        # print("##############################")
        # print(inputs)
        # print(pred_outputs)

        # keep track of previously seen output words
        prev_output_words = []

        for i, output in enumerate(pred_outputs):
            # print(inputs)
            # print(output)
            # print(i)

            # split current output into words
            output_words = output.strip().split()

            # get new words compared to previous step
            new_words = output_words[len(prev_output_words):]

            # count how many input words were read
            delay = len(inputs[i].strip().split())

            # assign current delay to each new word
            delays.extend([delay] * len(new_words))

            # update previous output words
            prev_output_words = output_words

        self.delays.append(delays)

        # computes AL scores for the input and output and stores the results
        self.call_AL_compute(inputs, gold_text, delays)

    def call_AL_compute(self, inputs, gold_text, delays):
        ALs = []

        # compute latency score using current delays
        source_len = len(inputs[-1].strip().split())
        target_len = len(gold_text)

        # calc AL only if there are calculated delays
        if len(delays) > 0:
            AL = compute(delays, source_len, target_len)
            ALs.append(AL)

        self._AL.append(ALs)

        # # prints to be deleted
        # print(f"Current AL: {self._AL}")
        # print("##############################")

    def calc_sacreBLEU(self):

        # compute BLEU
        bleu = sacrebleu.corpus_bleu(self.predictions, self.golden_trans)

        # print(f"bleu: {bleu.score}")
        self.bleu = bleu.score

    def eval(self):
        return {
            "delay_words": self.words__latency,
            # "delay_chars": self.chars__latency,
            # "delay_tokens": self.tokens_latency,
        }
