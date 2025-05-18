import json
import os

def extract_alignatt_data(json_file):
    """
    Extract data from AlignAtt approach JSON files. 
    Returns a list of dictionaries containing attention_frame_size, layers, and metrics.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    extracted_data = []
    for record in data:
        # Extract the required fields
        entry = {
            'attention_frame_size': record['args']['attention_frame_size'],
            'layers': record['args']['layers'],
            'metrics': record['all_metrics']
        }
        extracted_data.append(entry)
    
    return extracted_data


def extract_local_agreement_data(json_file):
    """
    Extract data from AlignAtt approach JSON files.
    Returns a list of dictionaries containing wait_for_begginning, beams, and metrics.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    extracted_data = []
    for record in data:
        # Extract the required fields
        entry = {
            'wait_for_beginning': record['args']['wait_for_beginning'],
            'num_beams': record['args']['num_beams'],
            'metrics': record['all_metrics']
        }
        extracted_data.append(entry)
    
    return extracted_data

def normalize_axis_value(val):
    # If it's a list or tuple of length 1, return the first element, else return as is
    if isinstance(val, (list, tuple)) and len(val) == 1:
        return val[0]
    return val

def create_metric_table(data, axis1, axis2, axis1_name, axis2_name, experiment_name=None, return_md=False):
    """
    Create a 2D table (nested dict) from the extracted data.
    axis1, axis2: strings, the keys to use for the axes (e.g., 'attention_frame_size', 'layers', 'num_beams', 'wait_for_beginning')
    axis1_name, axis2_name: pretty names for printing
    experiment_name: if provided, will be used as a header in the markdown file
    return_md: if True, return the markdown string instead of writing to file
    """
    axis1_values = sorted(set(normalize_axis_value(entry[axis1]) for entry in data))
    axis2_values = sorted(set(normalize_axis_value(entry[axis2]) for entry in data))

    table = {a1: {a2: None for a2 in axis2_values} for a1 in axis1_values}
    bleu_list, wer_list, al_list = [], [], []
    for entry in data:
        a1 = normalize_axis_value(entry[axis1])
        a2 = normalize_axis_value(entry[axis2])
        table[a1][a2] = entry['metrics']
        cell = entry['metrics']
        if cell:
            bleu_list.append(cell['bleu'])
            wer_list.append(cell['wer'])
            al_list.append(cell['AL'])

    # Find max BLEU, min WER, min AL (rounded to 2 decimals for comparison)
    max_bleu = round(max(bleu_list), 2) if bleu_list else None
    print(max_bleu)
    min_wer = round(min(wer_list), 2) if wer_list else None
    min_al = round(min(al_list), 2) if al_list else None

    md_lines = []
    if experiment_name:
        md_lines.append(f"\n## {experiment_name}\n")
    # Header row
    header = f"| {axis1_name} |" + "|".join(f"{str(a2)} ({axis2_name})" for a2 in axis2_values) + "|"
    md_lines.append(header)
    # Separator row
    md_lines.append("|" + "----|" * (len(axis2_values) + 1))
    # Data rows
    for a1 in axis1_values:
        row = [f"{a1}"]
        for a2 in axis2_values:
            cell = table[a1][a2]
            if cell:
                bleu_val = round(cell['bleu'], 2)
                wer_val = round(cell['wer'], 2)
                al_val = round(cell['AL'], 2)
                bleu = f"**{bleu_val:.2f}**" if bleu_val == max_bleu else f"{bleu_val:.2f}"
                wer = f"**{wer_val:.2f}**" if wer_val == min_wer else f"{wer_val:.2f}"
                al = f"**{al_val:.2f}**" if al_val == min_al else f"{al_val:.2f}"
                row.append(f"BLEU: {bleu}, AL: {al}, WER: {wer}")
            else:
                row.append("-")
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")
    md_str = "\n".join(md_lines)
    if return_md:
        return md_str
    with open("RESULTS.md", "a", encoding="utf-8") as f:
        f.write(md_str)
    return table

def write_side_by_side_tables_md(data1, data2, axis1, axis2, axis1_name, axis2_name, exp1_name, exp2_name):
    """
    Write two tables side by side in RESULTS.md using HTML.
    """
    table1_md = create_metric_table(data1, axis1, axis2, axis1_name, axis2_name, experiment_name=exp1_name, return_md=True)
    table2_md = create_metric_table(data2, axis1, axis2, axis1_name, axis2_name, experiment_name=exp2_name, return_md=True)
    html = f"""
<table><tr>
<td style='vertical-align:top; padding-right: 30px;'>
{table1_md}
</td>
<td style='vertical-align:top;'>
{table2_md}
</td>
</tr></table>
"""
    with open("RESULTS.md", "a", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    # AlignNatt extraction
    first_alignatt = "../AlignAttOutputs/parsed/baseline_alignatt.json"
    alignatt_baseline = extract_alignatt_data(first_alignatt)
    create_metric_table(alignatt_baseline, axis1='attention_frame_size', axis2='layers', axis1_name='Frame Size', axis2_name='Layers', experiment_name='AlignAtt Baseline')

    first_alignatt = "../AlignAttOutputs/parsed/finetuned_alignatt.json"
    alignatt_finetuned = extract_alignatt_data(first_alignatt)
    #write_side_by_side_tables_md(alignatt_baseline, alignatt_finetuned, axis1='attention_frame_size', axis2='layers', axis1_name='Frame Size', axis2_name='Layers', exp1_name='AlignAtt Baseline', exp2_name='AlignAtt Finetuned')
    create_metric_table(alignatt_finetuned, axis1='attention_frame_size', axis2='layers', axis1_name='Frame Size', axis2_name='Layers', experiment_name='AlignAtt Finetuned')

    #Local agreement extraction
    first_local = "../AlignAttOutputs/parsed/baseline_local_agreement.json"
    data = extract_local_agreement_data(first_local)
    create_metric_table(data, axis1='num_beams', axis2='wait_for_beginning', axis1_name='#Beams', axis2_name='Wait At Begin', experiment_name='Local Agreement MNT Baseline')

    first_local = "../AlignAttOutputs/parsed/finetuned_local_agreement.json"
    data = extract_local_agreement_data(first_local)
    create_metric_table(data, axis1='num_beams', axis2='wait_for_beginning', axis1_name='#Beams', axis2_name='Wait At Begin', experiment_name='Local Agreement Finetuned')

    first_local = "../AlignAttOutputs/parsed/LLM_local_agreement.json"
    data = extract_local_agreement_data(first_local)
    create_metric_table(data, axis1='num_beams', axis2='wait_for_beginning', axis1_name='#Beams', axis2_name='Wait At Begin', experiment_name='Local Agreement LLM')

    #Baseline extraction
    first_local = "../AlignAttOutputs/parsed/finetuned_wait_beginning.json"
    data = extract_local_agreement_data(first_local)
    create_metric_table(data, axis1='num_beams', axis2='wait_for_beginning', axis1_name='#Beams', axis2_name='Wait At Begin', experiment_name='Finetuned')

    first_local = "../AlignAttOutputs/parsed/baseline_wait_beginning.json"
    data = extract_local_agreement_data(first_local)
    create_metric_table(data, axis1='num_beams', axis2='wait_for_beginning', axis1_name='#Beams', axis2_name='Wait At Begin', experiment_name='MNT Backbone')

    # Example for side-by-side tables:
    # first_alignatt = "../AlignAttOutputs/parsed/baseline_alignatt.json"
    # data1 = extract_alignatt_data(first_alignatt)
    # first_alignatt2 = "../AlignAttOutputs/parsed/finetuned_alignatt.json"
    # data2 = extract_alignatt_data(first_alignatt2)
    # write_side_by_side_tables_md(data1, data2, axis1='attention_frame_size', axis2='layers', axis1_name='Frame', axis2_name='Layer', exp1_name='AlignAtt Baseline', exp2_name='AlignAtt Finetuned')



