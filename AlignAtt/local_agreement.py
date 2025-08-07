def local_agreement(new_theory, previous_theory, stable_theory, args):
    stop = min(len(new_theory), len(previous_theory))
    for i in range(len(stable_theory), stop):
        if any([new_theory[j] != previous_theory[j] for j in range(i, min(stop, i + args.local_agreement_length))]):
            break
        stable_theory += [new_theory[i]]
    return stable_theory