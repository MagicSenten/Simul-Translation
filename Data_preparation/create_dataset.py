#!/usr/bin/env python3
import json
import regex
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file_path", default="all.cs-en.top10M.txt", type=str, help="Path to the downloaded dataset from the readme file. Should be .txt format.")
parser.add_argument("--cleaned_dataset_path", default="cleaned_dataset.jsonl", type=str, help="Path where to save the cleaned dataset. Should be .jsonl format.")
parser.add_argument("--prefix_dataset_path", default="prefixes_dataset.jsonl", type=str, help="Path where to save the cleaned dataset. Should be .jsonl format.")
parser.add_argument("--number_of_lines", default=10, type=int, help="Number of lines used from the downloaded dataset for the prefix dataset. The default is the whole file")
#TODO: this part make sure the argument below works
parser.add_argument("--tokenizer", default=None, type=str, help="Path to the tokenizer on huggingface. NOT IMPLEMENTED YET!")
parser.add_argument("--create_prefixes_by_aligment", default=False, action="store_true", help="Create the prefix dataset by using aligment tools. NOT IMPLEMENTED YET!")

class CreateDataset():
    @staticmethod
    def _remove_leading_non_alnum_unicode(s: str) -> str:
        """
        Strip off any leading characters that are not Unicode letters or numbers,
        including Czech diacritics.
        """
        return regex.sub(r'^[^\p{L}\p{N}]+', '', s)

    def prepare_data_from_file(self, input_file: str = "all.cs-en.top10M.txt", output_file: str = "cleaned_dataset.jsonl", number_of_lines: int = None):
        """
        Cleaning function that is build for the 'all.cs-en.top10M.txt' 
        format of dataset downloaded from https://ufallab.ms.mff.cuni.cz/~machacek/cs-en-de-training-data/
        """
        pairs = []
        last_line = None
        with open(input_file, 'r', encoding='utf-8') as infile:
            for index, line in enumerate(infile):
                #Limit if the user wants to use just a portion of the dataset
                if number_of_lines is not None:
                    if index > number_of_lines:
                        break

                cols = line.strip().split('\t')
                if len(cols) < 4:
                    continue  # skip malformed lines

                if last_line is not None and last_line == line:
                    #Inside the data there are sometimes multiple lines that are same, 
                    #which is the reason for this line
                    continue

                source = self._remove_leading_non_alnum_unicode(cols[2])
                target = self._remove_leading_non_alnum_unicode(cols[3])
                pairs.append({
                    "source": f"{source}",
                    "target": f"{target}"
                })

                last_line = line

        self.save_pairs(output_file, pairs, append=False)


    def get_prefixes_by(self, source: str, target: str, tokenizer=None):
        """
        Defaultly creates sentence prefixes based on the number of characters. 
        If a tokenizer is provided, it uses tokens instead of the characters.
        """
        source_words = source.split()
        target_words = target.split()
        if tokenizer:
            source_char_counts = [len(tokenizer.tokenize(word)) for word in source_words]
            target_char_counts = [len(tokenizer.tokenize(word)) for word in target_words]
        else:
            source_char_counts = [len(word) for word in source_words]
            target_char_counts = [len(word) for word in target_words]
        
        
        #Decide which sentence is shorter
        length_proportions = sum(source_char_counts) / sum(target_char_counts)
        #Source is longer than the target
        if length_proportions > 1:
            long_words, long_counts = source_words, source_char_counts
            short_words, short_counts = target_words, target_char_counts
            source_is_long = True
        #Same proportion, finding the longer word by which to create the prefixes
        elif length_proportions == 1:
            shorter_sentence_length = np.minimum(len(source_words),len(target_words))
            found_longer_word = False
            for i in range(shorter_sentence_length):
                if source_char_counts[i] > target_char_counts[i]:
                    long_words, long_counts = target_words, target_char_counts 
                    short_words, short_counts = source_words, source_char_counts
                    source_is_long = False
                    found_longer_word = True
                    break
                elif source_char_counts[i] < target_char_counts[i]:
                    long_words, long_counts = source_words, source_char_counts
                    short_words, short_counts = target_words, target_char_counts
                    source_is_long = True
                    found_longer_word = True
                    break
                #The words in the i-th position have the same char/token length
                else:
                    continue
            #Same lenght sentences
            if not found_longer_word:
                    long_words, long_counts = target_words, target_char_counts 
                    short_words, short_counts = source_words, source_char_counts
                    source_is_long = False
        #Target is longer than the source
        else:
            long_words, long_counts = target_words, target_char_counts 
            short_words, short_counts = source_words, source_char_counts
            source_is_long = False

        #Sanity check before the indexing into the target_char_counts
        if len(long_counts) == 0:
            print(f"Invalid longer sentence inside get_prefixes_by function: {source}, {target}")
            return
        
        
        short_acc = 0
        long_acc = long_counts[0] 
        long_index = 1
        prefixes = []

        for index, cnt in enumerate(short_counts):
            #Shorter sentence finished first, create last pair and stop generating prefixes
            if index == len(short_words) - 1:
                prefixes.append((" ".join(short_words)," ".join(long_words)))
                break
            #Finding the prefix of the longer sentence that is shorter or equal by proportion of characters
            short_acc += cnt
            while long_index < len(long_counts) and long_acc + long_counts[long_index] <= length_proportions * short_acc:
                long_acc += long_counts[long_index]
                long_index += 1

            #Longer sentence finished first, create last prefix pair and stop generating prefixes
            if long_index == len(target_char_counts):
                prefixes.append((" ".join(short_words)," ".join(long_words)))
                break

            #Check if the longer sequence isn't inside the list already, if so, delete it.
            #We don't want to have any same prefixes apear multiple times inside the dataset
            #with different prefixes. That could mess up the finetuning
            current_long_prefix = " ".join(long_words[:long_index])
            if len(prefixes) > 0:
                _, last_added_long = prefixes[-1]
                if current_long_prefix == last_added_long:
                    prefixes.pop()

            #Append the sentence prefix pair
            prefixes.append((" ".join(short_words[:index+1]),
                            current_long_prefix))

        #Invert the position of the pairs
        if source_is_long:
            prefixes = [(src, tgt) for tgt, src in prefixes]


        return [{"pref_source": src_pref, "pref_target": tgt_pref}
                for src_pref, tgt_pref in prefixes]

    def get_prefixes_by_alligment():
        ''''''
        #TODO: make this function
        ...

    def save_pairs(self, file_name: str, pairs, append: bool = True):
        """
        Saves the created pairs into the file_name via json dump. Defaultly appends to the file.
        """
        mode='a'
        if not append:
            mode = 'w'
        with open(file_name, mode, encoding='utf-8') as outfile:
            for item in pairs:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')


    def create_prefixes(self, cleaned_dataset_file_path: str = "cleaned_dataset.jsonl", prefix_dataset_file_path: str = "prefixes_dataset.jsonl", tokenizer=None):
        """
        Generate incremental prefix pairs for each source-target example in a prepared JSONL file.
        An optional `tokenizer` can be provided for token-based prefix segmentation.

        :param cleaned_dataset_file_path: Path to the input JSONL file with full source/target pairs.
        :param prefix_dataset_file_path: Path to the output JSONL file for storing prefix pairs.
        :param tokenizer: Optional tokenizer for token-based splitting; defaults to None.
        """
        with open(cleaned_dataset_file_path, 'r', encoding='utf-8') as f:
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

                #For each line create the prefixes
                src = obj["source"]
                tgt = obj["target"]
                prefix_pairs = self.get_prefixes_by(src, tgt, tokenizer)
                prefix_pairs_collection.extend(prefix_pairs)
                #Periodic save
                if lineno % 100 == 0:
                    self.save_pairs(prefix_dataset_file_path, prefix_pairs_collection)
                    prefix_pairs_collection.clear()

        #Final save
        self.save_pairs(prefix_dataset_file_path, prefix_pairs_collection)



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    dataset = CreateDataset()
    dataset.prepare_data_from_file(input_file=main_args.input_file_path, output_file=main_args.cleaned_dataset_path, number_of_lines=main_args.number_of_lines)
    #TODO: add the tokenizer path
    #TODO: implement the aligment branch here
    dataset.create_prefixes(cleaned_dataset_file_path=main_args.cleaned_dataset_path, prefix_dataset_file_path=main_args.prefix_dataset_path)
