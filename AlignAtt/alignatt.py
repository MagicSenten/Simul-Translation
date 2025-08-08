import numpy as np

def visualize_attention(input_ids, output_ids, attentions, tokenizer, args):
    """
    Visualizes attention weights by extracting and decoding the most attended tokens.

    Args:
        input_ids (list): A list of input token IDs.
        output_ids (list): A list of output token IDs.
        attentions (list): A list of attention tensors with shape
            (batch_size, num_heads, generated_length, sequence_length).
        tokenizer: The tokenizer used to decode token IDs into text.
        args (argparse.Namespace): Parsed command-line arguments containing attributes
            such as `heads` (list of attention heads).

    Prints:
        - Top attention positions for the last few output tokens.
        - Corresponding decoded tokens for the attention positions.
        - Decoded output tokens for different segments of the output.
    """
    def sort_top(l, t):
        """
        Sorts the top attention positions, keeping the last `t` positions in order.

        Args:
            l (list): A list of attention positions.
            t (int): The number of positions to keep in order.

        Returns:
            list: Sorted attention positions with the last `t` positions preserved.
        """
        return [y - len(input_ids) for y in l[:-t] + sorted(l[-t:])]

    def get_range(vs):
        """
        Decodes a range of input tokens based on attention positions.

        Args:
            vs (list): A list of attention positions.

        Returns:
            str: Decoded text for the specified range of input tokens.
        """
        if len(vs) < 3:
            return ""
        ids = input_ids[min(vs[-3:]):max(vs[-3:])+1]
        r = tokenizer.decode(ids)
        return r

    top = 10
    last = 10
    # Get the top attention positions for the last 5 output tokens
    print([sort_top(y[0, args.heads, -1, :].mean(0).argsort(-1)[-top:].tolist(), 3) for y in attentions[-last:]])
    # Print the corresponding tokens
    print([get_range(x[0, args.heads, -1, :].mean(0).argsort(-1)[-top:].tolist()) for x in attentions[:last]])
    print(tokenizer.decode(output_ids[:last]))
    print([get_range(x[0, args.heads, -1, :].mean(0).argsort(-1)[-top:].tolist()) for x in attentions[-last:-last//2]])
    print(tokenizer.decode(output_ids[-last:-last//2]))
    print([get_range(x[0, args.heads, -1, :].mean(0).argsort(-1)[-top:].tolist()) for x in attentions[-last//2:]])
    print(tokenizer.decode(output_ids[-last//2:]))


def alignatt(attentions, args):
    """
    Aligns attention weights by identifying the first position where the top attention
    positions meet specific criteria.

    Args:
        attentions (list): A list of attention tensors with shape
            (batch_size, num_heads, generated_length, sequence_length).
        args (argparse.Namespace): Parsed command-line arguments containing attributes
            such as `heads`, `top_attentions`, `skip_l`, `attention_frame_size`, and `count_in`.

    Returns:
        int: The index of the first attention tensor meeting the criteria, or the total
        number of attention tensors if no match is found.
    """
    for i in range(len(attentions)):
        # Compute mean attention weights for the specified heads
        mean_attentions = attentions[i][0, args.heads, -1, :].mean(0)
        # Get the top attention positions
        top_pos = mean_attentions.argsort(-1)[-args.top_attentions:].cpu().numpy()
        # Exclude positions within the skip range
        top_pos[top_pos >= attentions[0].shape[-1] - args.skip_l] = 0
        # Check if the top positions meet the alignment criteria
        if np.sum(np.less_equal(attentions[0].shape[-1] - args.attention_frame_size, top_pos)) >= args.count_in:
            print(i, len(attentions), attentions[0].shape[-1] - top_pos)
            return i
    print(len(attentions), "full", attentions[0].shape[-1] - top_pos)
    return len(attentions)