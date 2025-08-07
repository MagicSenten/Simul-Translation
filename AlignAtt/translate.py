import torch
from transformers import GenerationConfig, PreTrainedTokenizerBase

from alignatt import alignatt, visualize_attention


def translate_LLM(model, tokenizer, input_text, stable_theory, args, computation_stats, verbose=False):
    '''
        - 'prefix' Refers to a substring, for each substring.
        - 'pt': Return as pytorch tensor.
    '''
    is_sent_end = input_text.endswith(".")
    decoder_input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(stable_theory)).unsqueeze(0) if len(stable_theory) > 0 else None

    input_text = f'<|im_start|>system\nYou are simultaneous interpreter from Czech to English, you translate incomplete sentences, please make sure you only translate what is explicitly stated in the input segment.<|im_end|>\n<|im_start|>user\nTranslate the following Czech source text to English.\nCzech: {input_text}\nEnglish: <|im_end|>\n<|im_start|>assistant\n'
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    all_input_ids = torch.cat([input_ids, decoder_input_ids], 1) if decoder_input_ids is not None else input_ids
    """
      cross_attentions (tuple(tuple(torch.FloatTensor)), optional,
      returned when output_attentions=True) —
      Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor
      of shape (batch_size, num_heads, generated_length, sequence_length).
    """
    bad_words = ["English:", "Czech:", "<0x0A>", "Reference:"]
    config = GenerationConfig(num_beams=args.num_beams, num_beam_groups=args.num_beams//3 if args.num_beams % 3 == 0 else 1, diversity_penalty=0.1 if args.num_beams % 3 == 0 and args.num_beams > 3 else 0, no_repeat_ngram_size=2,
                              length_penalty=0.98 if args.num_beams > 1 else 1.0, bad_words_ids= [tokenizer.encode(x) for x in bad_words])
    outputs = model.generate(input_ids=all_input_ids.to(args.device), generation_config=config,
                             return_dict_in_generate=True, output_attentions=True, max_new_tokens=min(2, input_ids.shape[1]*1.5-len(stable_theory)), renormalize_logits=True, forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.forced_bos_token_text) if args.forced_bos_token_text is not None else None, pad_token_id=tokenizer.pad_token_id)

    ca = outputs["attentions"]
    len_output_ids = len(outputs["sequences"][0])
    output_ids = outputs["sequences"][0][decoder_input_ids.shape[1]+1 if decoder_input_ids is not None else 0:].cpu()
    if args.top_attentions > 0 and not decoder_input_ids is None and len(ca[1:]) > 0:
        print([x[0].shape for x in ca[1:]])
        attentions = [sum(x[-1-i][:, :, :, all_input_ids.shape[1]:] for i in args.layers) for x in ca[1:]]
        attentions = attentions[:len(output_ids) - decoder_input_ids.shape[1]]
        if verbose:
            visualize_attention(input_ids[0], output_ids[decoder_input_ids.shape[1]:], attentions, tokenizer, args)
        alignatt_result = alignatt(attentions, args)
    else:
        alignatt_result = len(output_ids)
    decoded_align_att = tokenizer.convert_ids_to_tokens(output_ids[:alignatt_result], skip_special_tokens=True)
    return decoded_align_att

def translate(model, tokenizer: PreTrainedTokenizerBase, input_text, stable_theory, computation_stats, args, verbose=False):
    '''
        - 'prefix' Refers to a substring, for each substring.
        - 'pt': Return as pytorch tensor.
    '''
    is_sent_end = input_text.endswith(".")
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    decoder_input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(stable_theory)).unsqueeze(0) if len(stable_theory) > 0 else None
    """
      cross_attentions (tuple(tuple(torch.FloatTensor)), optional,
      returned when output_attentions=True) —
      Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor
      of shape (batch_size, num_heads, generated_length, sequence_length).
    """
    config = GenerationConfig(num_beams=args.num_beams, num_beam_groups=args.num_beams//3 if args.num_beams % 3 == 0 else 1, diversity_penalty=0.1 if args.num_beams % 3 == 0 and args.num_beams > 3 else 0, no_repeat_ngram_size=2,
                              length_penalty=0.98 if args.num_beams > 1 else 1.0)
    outputs = model.generate(input_ids=input_ids.to(args.device), decoder_input_ids=decoder_input_ids.to(
        args.device) if decoder_input_ids is not None else None,
                             return_dict_in_generate=True, output_attentions=True, max_new_tokens=10,
                             generation_config=config, renormalize_logits=True, forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.forced_bos_token_text) if args.forced_bos_token_text is not None else None)
    ca = outputs["cross_attentions"]
    print(len(ca), len(ca[0]), len(ca[0][0]), len(ca[0][0][0]))
    outputsequence = outputs["sequences"][0].cpu()
    print(tokenizer.convert_ids_to_tokens(outputsequence))
    if False:
        print(tokenizer.convert_ids_to_tokens(decoder_input_ids[0]))
        assert len(ca) - 2 == len(outputsequence) - decoder_input_ids.shape[1] - 1, f"or {len(ca)} {len(outputsequence)} {decoder_input_ids.shape[1]}"
    if outputsequence[-1] == tokenizer.eos_token_id:
        outputsequence = outputsequence[:-1]
    len_output = len(outputsequence)
    output_ids = outputsequence[decoder_input_ids.shape[1] if decoder_input_ids is not None else 0:].cpu()
    if args.top_attentions > 0 and not decoder_input_ids is None and len(ca) > 1:
        assert all([x[0].shape[2] == 1 for x in ca[1:]])
        #assert len(ca) == len(output_ids), f"or {len(ca)} {len(output_ids)}"
        attentions = [sum(x[-1-i][:1, :, -1:] for i in args.layers) for x in ca]
        if verbose:
            visualize_attention(input_ids[0], output_ids, attentions, tokenizer, args)
        alignatt_relative = alignatt(attentions, args)
        #If for some reason the attentions are too short remove from allignatt result
        extra_length =  (len_output - decoder_input_ids.shape[1]) - len(attentions)
        if extra_length > 0:
            computation_stats["attentions_too_short{extra_length}"] = computation_stats.get("attentions_too_short", 0) + 1
            alignatt_relative = alignatt_relative - extra_length

        alignatt_relative = max(0, alignatt_relative)
        alignatt_is_zero = alignatt_relative == 0
        computation_stats["total_its"] = computation_stats.get("total_its", 0) + 1
        if alignatt_is_zero:
            computation_stats["alignatt_is_zero"] = computation_stats.get("alignatt_is_zero", 0) + 1
        if alignatt_is_zero and len(input_text.split(" ")) % 4 == 0:
            alignatt_relative = 1
        alignatt_result = alignatt_relative
    else:
        alignatt_result = len(output_ids)

    decoded_align_att = tokenizer.convert_ids_to_tokens(output_ids[:alignatt_result], skip_special_tokens=False)
    return decoded_align_att