from typing import List, Union
from jiwer import wer
import sacrebleu
import numpy as np

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
        """
        Initializes the SimuEval object.

        Attributes:
            delays (List[List[Union[float, int]]]): A list to store sequences of delays for each processed sentence.
            predictions (List[str]): A list to store the final predicted translations for each sentence.
            golden_trans (List[str]): A list to store the ground truth translations for each sentence.
            _AL (List[List[float]]): A list to store Average Lagging scores for each sentence.
                                     Each inner list contains AL scores (typically one per sentence).
            WERs (List[float]): A list to store Word Error Rate (WER) scores for each sentence.
            bleu (float): The corpus-level BLEU score.
            avg_WER (float): The average Word Error Rate over all sentences.
        """
        self.delays = []

        self.predictions = []
        self.golden_trans = []

        self._AL = []
        self.bleu = 0
        self.WERs = []
        self.avg_WER = 0

    def update(self, inputs, pred_outputs, gold_text):
        """
        Updates the evaluation metrics with a new instance of inputs, predicted outputs, and gold text.
        It calculates per-word delays, stores predictions and gold text for quality evaluation,
        and computes the AL score for the current instance.

        Args:
            inputs (List[str]): List of source inputs at different time steps.
                                `inputs[i]` is the source text available when `pred_outputs[i]` was generated.
                                `inputs[-1]` is the full source text.
            pred_outputs (List[str]): List of predicted outputs (translations) at different time steps.
                                      `pred_outputs[i]` is the translation prefix generated based on `inputs[i]`.
                                      `pred_outputs[-1]` is the final translation.
            gold_text (str): The true translated sentence.

        Returns:
            List[List[Union[float, int]]]: The `self.delays` attribute, which is a list containing
                                           lists of delays for each processed instance. The last appended
                                           inner list corresponds to the current call's calculated delays.

        Modifies:
            self.predictions: Appends the final predicted output (`pred_outputs[-1]`).
            self.golden_trans: Appends the `gold_text`.
            self.delays: Appends a list of calculated delays for the current instance.
            self._AL: Appends the AL score for the current instance (via `call_AL_compute`).
        """
        delays = []

        # save the data for the sacreBLEU evaluation
        self.predictions.append(pred_outputs[-1])
        self.golden_trans.append(gold_text)

        # keep track of previously seen output words
        prev_output_words = []

        for i, output in enumerate(pred_outputs):
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

        return self.delays

    def call_AL_compute(self, inputs, gold_text, delays):
        """
        Computes the Average Lagging (AL) score for the current instance using the provided delays
        and stores it in the `_AL` attribute.

        Args:
            inputs (List[str]): List of source inputs at different time steps.
                                `inputs[-1]` (the full source text) is used to determine source length.
            gold_text (str): The ground truth target sentence, used to determine target length.
            delays_current_instance (List[Union[float, int]]): Sequence of delays calculated for the current instance.

        Modifies:
            self._AL: Appends a list containing the computed AL score for the current instance.
                      If no delays are present, an empty list is appended for this instance's AL.
        """
        ALs = []

        # compute latency score using current delays
        source_len = len(inputs[-1].strip().split())
        target_len = len(gold_text)

        # calc AL only if there are calculated delays
        if len(delays) > 0:
            AL = compute(delays, source_len, target_len)
            ALs.append(AL)

        self._AL.append(ALs)

    def calc_WER(self):
        """
        Calculate average Word Error Rate (WER) over lists of golden texts and predictions.
        Saves the results both individually and an average result.

        Args:
            None (uses `self.golden_trans` and `self.predictions`).

        Returns:
            Tuple[List[float], float]: A tuple containing:
                - WERs (List[float]): List of the WER calculated for each golden texts and predictions pair.
                - avg_WER (float): Float representing the average WER score on all pairs.

        Modifies:
            self.WERs: Populated with individual WER scores for each sentence pair.
            self.avg_WER: Set to the calculated average WER.
        """
        total_wer = 0.0
        # clear previous results if any
        self.WERs = []
        n = len(self.golden_trans)
        for ref, hyp in zip(self.golden_trans, self.predictions):
            wer_res = wer(ref, hyp)
            self.WERs.append(wer_res)
            total_wer += wer_res

        self.avg_WER = total_wer / n if n > 0 else 0.0

        return self.WERs, self.avg_WER

    def calc_sacreBLEU(self):
        """
        Calculates the BLEU score using sacreBLEU based on the stored
        predictions and golden translations.

        Args:
            None (uses `self.predictions` and `self.golden_trans`).

        Returns:
            float: The corpus BLEU score. Returns 0.0 if no predictions or references.

        Modifies:
            self.bleu: Set to the calculated BLEU score.
        """
        # compute BLEU
        bleu = sacrebleu.corpus_bleu(self.predictions, [[x] for x in self.golden_trans])

        self.bleu = bleu.score
        return bleu.score

    def eval(self):
        return {
            "bleu": self.calc_sacreBLEU(),
            "wer": self.calc_WER()[1],
            "AL": np.mean(self._AL),
        }