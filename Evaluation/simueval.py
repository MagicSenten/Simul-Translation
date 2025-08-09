from typing import List, Union, Dict, Any, Tuple
from jiwer import wer
import sacrebleu
import json
from pyarrow.ipc import open_file


class SimuEval:
    def __init__(self):
        """
        Initializes the SimuEval object.

        Attributes:
            delays (List[List[Union[float, int]]]): A list to store sequences
             of delays for each processed sentence.

            predictions (List[str]): A list to store the final predicted
             translations for each sentence.

            golden_trans (List[str]): A list to store the ground truth
             translations for each sentence.

            _AL (List[List[float]]): A list to store Average Lagging
             scores for each sentence. Each inner list contains AL
             scores (typically one per sentence).

            TERs (List[float]): A list to store Word Error
            Rate (TER) scores for each sentence.

            bleu (float): The corpus-level BLEU score.
            avg_TER (float): The average Word Error Rate over all sentences.
        """
        self.delays = []

        self.predictions = []
        self.golden_trans = []

        self._AL = []
        self.bleu = -1
        self.TERs = []
        self.avg_TER = -1

    def process_data(self, data_dict: Dict) -> None:
        """
        Processes a batch of evaluation data by extracting inputs, predicted
        outputs, and gold texts from the provided dictionary, and updating
        the evaluation metrics for each instance.

        Args:
           data_dict (dict): A dictionary containing the evaluation
           data with the structure: {
           "data": {
                "inputs" (List[List[str]]): Source inputs at various time
                steps for each instance.

               "outputs" (List[List[str]]): Predicted outputs at various time
               steps for each instance.

               "texts" (List[str]): Ground truth target sentences
               for each instance. }
            }

        Returns:
           None

        Modifies:
           Calls `self.update` for each set of (inputs, predicted
           outputs, gold text), thereby updating:
           - self.predictions
           - self.golden_trans
           - self.delays
           - self._AL


        ### Example:
            from simueval import SimuEval
            import json

            # Initialize the evaluator
            evaluator = SimuEval()

            # Define a helper to read JSON data
            def read_json_to_dict(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            # Path to a JSON file containing the evaluation data
            file_path = r"<path_to_directory>/baseline_alignatt.json"

            # Load and process the data
            results = read_json_to_dict(file_path)
            evaluator.process_data(results[0])

            # Print evaluation metrics
            print(evaluator.eval())
        """
        inputs_list = data_dict["data"]["inputs"]
        pred_outputs_list = data_dict["data"]["outputs"]
        gold_text_list = data_dict["data"]["texts"]

        for inputs, pred_outputs, gold_text in zip(inputs_list,
                                                   pred_outputs_list,
                                                   gold_text_list):
            self.update(inputs, pred_outputs, gold_text)

    @staticmethod
    def compute_latency(
            delays: List[int],
            source_length: int,
            target_length: int,
    ):
        """
        Compute latency on one sentence (instance).

        Args:
            delays (List[Union[float, int]]): Sequence of delays.
            source_length (Union[float, int]): Length of source sequence.
            target_length (Union[float, int]): Length of target sequence.

        Returns:
            float: The latency score on one sentence.
        """
        if delays[0] > source_length:
            return delays[0]

        al = 0
        gamma = target_length / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            if d <= source_length:
                al += d - t_minus_1 / gamma
                tau = t_minus_1 + 1

                if d == source_length:
                    break
        al /= tau
        return al

    def update(
            self,
            inputs: List[str],
            pred_outputs: List[str],
            gold_text: str
    ) -> List[List[int]]:
        """
        Updates the evaluation metrics with a new instance of inputs,
        predicted outputs, and gold text.
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

    def call_AL_compute(
            self,
            inputs: List[str],
            gold_text: str,
            delays: List[int]
    ) -> None:
        """
        Computes the Average Lagging (AL) score for the current instance using the provided delays
        and stores it in the `_AL` attribute.

        Args:
            inputs (List[str]): List of source inputs at different time steps.
                                `inputs[-1]` (the full source text) is used to determine source length.
            gold_text (str): The ground truth target sentence, used to determine target length.
            delays (List[Union[float, int]]): Sequence of delays calculated for the current instance.

        Modifies:
            self._AL: Appends a list containing the computed AL score for the current instance.
                      If no delays are present, an empty list is appended for this instance's AL.
        """
        al_list = []

        # compute latency score using current delays
        source_len = len(inputs[-1].strip().split())
        target_len = len(gold_text)

        # calc AL only if there are calculated delays
        if len(delays) > 0:
            al = self.compute_latency(delays, source_len, target_len)
            al_list.append(al)

        self._AL.append(al_list)

    def calc_TER(self) -> Tuple[List[float], float]:
        """
        Calculate average Word Error Rate (TER) over lists of golden texts and predictions.
        Saves the results both individually and an average result.

        Returns:
            Tuple[List[float], float]: A tuple containing:
                - TERs (List[float]): List of the TER calculated for each golden texts and predictions pair.
                - avg_TER (float): Float representing the average TER score on all pairs.

        Modifies:
            self.TERs: Populated with individual TER scores for each sentence pair.
            self.avg_TER: Set to the calculated average TER.
        """
        total_ter = 0.0
        # clear previous results if any
        self.TERs = []
        n = len(self.golden_trans)
        for ref, hyp in zip(self.golden_trans, self.predictions):
            ter_res = wer(ref, hyp)
            self.TERs.append(ter_res)
            total_ter += ter_res

        self.avg_TER = total_ter / n if n > 0 else 0.0

        return self.TERs, self.avg_TER

    def calc_sacreBLEU(self) -> float:
        """
        Calculates the BLEU score using sacreBLEU based on the stored
        predictions and golden translations.

        Returns:
            float: The corpus BLEU score. Returns 0.0 if no predictions or references.

        Modifies:
            self.bleu: Set to the calculated BLEU score.
        """
        # compute BLEU
        bleu = sacrebleu.corpus_bleu(self.predictions, self.golden_trans)

        self.bleu = bleu.score

        return bleu.score

    def eval(self) -> Dict[str, float]:
        """
            Evaluates the stored predictions against the ground truth translations
            using BLEU, TER, and Average Lagging (AL) metrics.

            Returns:
                Dict[str, float]: A dictionary containing:
                    - "bleu" (float): The corpus BLEU score computed via `calc_sacreBLEU`.
                    - "ter" (float): The average Translation Error Rate computed via `calc_TER`.
                    - "AL" (float): The average lagging score computed over all stored instances.

            Modifies:
                self.bleu: Set to the calculated BLEU score.
                self.TERs: Populated with individual TER scores.
                self.avg_TER: Set to the average TER score.
                self._AL: Used to compute the average AL score.
            """
        return {
            "bleu": self.calc_sacreBLEU(),
            "ter": self.calc_TER()[1],
            "AL": sum([x[-1] for x in self._AL]) / len(self._AL),
        }
