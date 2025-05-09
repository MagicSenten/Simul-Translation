import numpy as np
def visualize_attention(input_ids, output_ids, attentions, tokenizer, args):
    def sort_top(l, t):
        return [y - len(input_ids) for y in l[:-t] + sorted(l[-t:])]

    def get_range(vs):
        if len(vs) < 3:
            return ""
        ids = input_ids[min(vs[-3:]):max(vs[-3:])]
        r = tokenizer.decode(ids)
        return r

    # get the top attention positions for the last 5 output tokens (-1 means last input token)
    print([sort_top(y[0, args.heads, -1, :].mean(0).argsort(-1)[-10:].tolist(), 3) for y in attentions[:-10]])
    # print the corresponding tokens
    print([get_range(x[0, args.heads, -1, :].mean(0).argsort(-1)[-10:].tolist()) for x in attentions[-10:-5]])
    print(tokenizer.decode(output_ids[-10:-5]))
    print([get_range(x[0, args.heads, -1, :].mean(0).argsort(-1)[-10:].tolist()) for x in attentions[:-5]])
    print(tokenizer.decode(output_ids[:-5]))


def alignatt(attentions, args):
    for i in range(len(attentions)):
        # shape (batch_size, num_heads, generated_length, sequence_length)
        mean_attentions = attentions[i][0, args.heads, -1, :].mean(0)
        # shape (generated_length)
        top_pos = mean_attentions.argsort(-1)[-args.top_attentions:].cpu().numpy()
        top_pos[top_pos >= attentions[0].shape[-1] - args.skip_l] = 0
        # print(attentions[i].shape, top_pos, mean_attentions[-mean_attentions.shape[0]//8:])
        if np.sum(np.less_equal(attentions[0].shape[-1] - args.attention_frame_size, top_pos)) > args.count_in:
            print(i, len(attentions), attentions[0].shape[-1] - top_pos)
            return i
    print(len(attentions), "full", attentions[0].shape[-1] - top_pos)
    return len(attentions)