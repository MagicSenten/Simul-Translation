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

def create_metric_table(data, axis1, axis2, axis1_name, axis2_name, experiment_name=None):
    """
    Create a 2D table (nested dict) from the extracted data.
    axis1, axis2: strings, the keys to use for the axes (e.g., 'attention_frame_size', 'layers', 'num_beams', 'wait_for_beginning')
    axis1_name, axis2_name: pretty names for printing
    experiment_name: if provided, will be used as a header in the markdown file
    """
    # Collect all unique values for each axis, normalized
    axis1_values = sorted(set(normalize_axis_value(entry[axis1]) for entry in data))
    axis2_values = sorted(set(normalize_axis_value(entry[axis2]) for entry in data))

    # Build the table as a nested dict: table[axis1][axis2] = metrics
    table = {a1: {a2: None for a2 in axis2_values} for a1 in axis1_values}
    for entry in data:
        a1 = normalize_axis_value(entry[axis1])
        a2 = normalize_axis_value(entry[axis2])
        table[a1][a2] = entry['metrics']

    # Print the table
    print_table = False
    if print_table:
        print(f"\nTable: {axis1_name} (rows) vs {axis2_name} (columns)")
        header = f"{axis1_name:>10} |" + "|".join(f" {str(a2):>8} " for a2 in axis2_values)
        print(header)
        print("-" * len(header))
        for a1 in axis1_values:
            row = f"{str(a1):>10} |"
            for a2 in axis2_values:
                cell = table[a1][a2]
                if cell:
                    row += f" BLEU: {cell['bleu']:.2f}, WER: {cell['wer']:.2f}, AL: {cell['AL']:.2f} |"
                else:
                    row += f" {'-':>18} |"
            print(row)
        print()

    # Save as markdown to RESULTS.md
    md_lines = []
    if experiment_name:
        md_lines.append(f"\n## {experiment_name}\n")
    md_lines.append(f"| {axis1_name} \\ {axis2_name} |" + "|".join(f" {str(a2)} " for a2 in axis2_values) + "|")
    md_lines.append("|" + ":---:|" * (len(axis2_values)+1))
    for a1 in axis1_values:
        row = [f"{a1}"]
        for a2 in axis2_values:
            cell = table[a1][a2]
            if cell:
                row.append(f"BLEU: {cell['bleu']:.2f}<br>WER: {cell['wer']:.2f}<br>AL: {cell['AL']:.2f}")
            else:
                row.append("-")
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")
    with open("RESULTS.md", "a", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return table

if __name__ == "__main__":
    # AlignNatt extraction
    first_alignatt = "../AlignAttOutputs/parsed/baseline_alignatt.json"
    data = extract_alignatt_data(first_alignatt)
    create_metric_table(data, axis1='attention_frame_size', axis2='layers', axis1_name='Frame Size', axis2_name='Layers', experiment_name='AlignAtt Baseline')

    first_alignatt = "../AlignAttOutputs/parsed/finetuned_alignatt.json"
    data = extract_alignatt_data(first_alignatt)
    create_metric_table(data, axis1='attention_frame_size', axis2='layers', axis1_name='Frame Size', axis2_name='Layers', experiment_name='AlignAtt Finetuned')

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



