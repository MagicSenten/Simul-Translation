from typing import List, Union
# added comment for commit

def calc_latency_ratio(partial, full, pred, gold):
    return (len(gold) / len(full) * len(partial) - len(pred)) / len(gold)


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
        self.gold_text = ""
        self.cur_pred_pos = 0

        self.delays = []

        self.latency_scorer = []
        self.quality_scorer = []

        self.words__latency = 0
        # self.chars__latency = 0
        # self.tokens_latency = 0

        self.cs = 0

    def update(self, inputs, pred_outputs, gold_text, tokenizer):
        print(f"inputs: {inputs}")
        # init current delay list
        cur_delay = []

        # the length of the inputs and output
        prefix_len = len(inputs)

        for i in range(prefix_len):
            pred_len = len(pred_outputs[i].strip().split(" "))
            for _ in range(pred_len):
                inp_len = len(inputs[i].strip().split(" "))
                cur_delay.append(inp_len)

            self.cur_pred_pos = pred_len
        self.delays.append(cur_delay)

        # # marks the new iteration of a new gold_label
        # if self.gold_text != gold_text:
        #     '''
        #         init for each pair:
        #         - save the delay list for that pair in the delays list
        #         - init cur_delay list
        #         - init cur_pred_pos
        #     '''
        #     self.gold_text = gold_text
        #     # only append non empty lists
        #     if len(self.cur_delay) > 0:
        #         self.delays.append(self.cur_delay)
        #     self.cur_delay = []
        #     self.cur_pred_pos = 0
        #
        # # calcs how many words were predicted, makes sure to clean trailing spaces before splitting
        # pred_len = len(predicted_text.strip().split(" "))
        # # calcs the position we had to read to in input to get the stabalized output
        # part_len = len(partial_input_text.split(" "))
        #
        # for i in range(self.cur_pred_pos, pred_len):
        #     self.cur_delay.append(part_len)
        #
        # # update the cur position to be read to
        # self.cur_pred_pos = pred_len
        #
        # # returns the updated delays list
        # return self.delays
        return None

    def eval(self):
        return {
            "delay_words": self.words__latency,
            # "delay_chars": self.chars__latency,
            # "delay_tokens": self.tokens_latency,
        }
