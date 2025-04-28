#!/usr/bin/env python3
import json


class CreateDataset():

    def prepare_data_from_file(self, input_file: str = "all_cs-en_top10M.txt", output_file: str = "dataset.jsonl"):
        '''Cleaning function that is build for the 'all.cs-en.top10M.txt' format'''
        pairs = []
        last_line = None
        with open(input_file, 'r', encoding='utf-8') as infile:
            for index, line in enumerate(infile):
                if index > 10:
                    break
                cols = line.strip().split('\t')
                if len(cols) < 4:
                    continue  # skip malformed lines

                if last_line is not None and last_line == line:
                    continue

                source = cols[2]
                target = cols[3]
                pairs.append({
                    "source": f"{source}",
                    "target": f"{target}"
                })

                last_line = line

        self.save_pairs(output_file, pairs, append=False)


    def get_prefixes_by_chars(self, source: str, target: str, fraction: float):
        source_words = source.split()
        target_words = target.split()
        src_acc = 0
        prefixes = []
        for source_word in source_words:
            src_acc += len(source_word)
            tgt_acc = 0
            tgt_pref_words

        ...

    def get_prefixes_by_tokens(self,source: str, target: str, tokenizer, fraction: float):
        ...

    def get_prefixes_by_alligment():
        ''''''
        ...

    def save_pairs(self, file_name: str, pairs, append: bool = True):
        '''Saves the created pairs into the file_name via json dump. Defaultly appends to the file.'''

        mode='a'
        if not append:
            mode = 'w'

        with open(file_name, mode, encoding='utf-8') as outfile:
            for item in pairs:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')


    def create_prefixes(self, prepared_data_file: str, prefix_dataset_file_name: str, fraction: float,
                        tokenizer=None):
        
        with open(prepared_data_file, 'r', encoding='utf-8') as f:
            prefix_pairs_collection = []
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Line {lineno} is malformed: {e}")
                    continue

                #For each line create the prefixes according to the chosen way
                src = obj["source"]
                tgt = obj["target"]
                if tokenizer is None:
                    prefix_pairs = self.get_prefixes_by_chars(src, tgt, fraction)
                else:
                    prefix_pairs = self.get_prefixes_by_tokens(src, tgt, tokenizer, fraction)

                prefix_pairs_collection.append(prefix_pairs)

                #Periodic save
                if lineno % 100 == 0:
                    self.save_pairs(prefix_dataset_file_name, prefix_pairs_collection)
                    prefix_pairs_collection.clear()

        #Final save
        self.save_pairs(prefix_dataset_file_name, prefix_pairs_collection)



if __name__ == "__main__":
    dataset = CreateDataset()
    dataset.prepare_data_from_file()
