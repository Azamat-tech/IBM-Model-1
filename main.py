import argparse
import numpy as np
import string

from string import punctuation

class Corpus: 
    def __init__(self, args, name="data/czenali") -> None:
        self.punctuation = args.punctuation_allowed
        self.file_location = name

        self.unique_eng = {}
        self.unique_cz  = {}

        self.reverse_unique_eng = {}

        self.eng_index = 0
        self.cz_index  = 0
        
        self.store(name)
        
        self.reverse_eng_dictionary()
    
    def store(self, name): 
        """
            Stores the eng/cz words into corresponding dictionary as a key
            and their index and their value
        """
        line_count = 0
        with open(name, "r", encoding="utf8") as file:
            for line in file:
                if args.to_lower:
                    line = line.lower()

                columns = line.split('\t')
                self.store_tokens(columns[0].split(), english=True)
                self.store_tokens(columns[1].split(), english=False)

                line_count += 1

                if line_count == args.sentences:
                    break

    def reverse_eng_dictionary(self): 
        """
            reverse english dictionary to access the english word 
            based on its index
        """
        for key in self.unique_eng:
            self.reverse_unique_eng[self.unique_eng[key]] = key

    def store_tokens(self, sentence, english):
        """
            stores words to the dictionary
        """
        for word in sentence:
            if is_punctuation(self.punctuation, word):
                continue
                
            if english: 
                if word not in self.unique_eng:
                    self.unique_eng[word] = self.eng_index
                    self.eng_index += 1
            else: 
                if word not in self.unique_cz:
                    self.unique_cz[word] = self.cz_index
                    self.cz_index += 1
    
class WordAlignment:
    def __init__(self, eng_size, cz_size, args) -> None:
        self.lower = args.to_lower
        self.punctuation = args.punctuation_allowed
        self.sentences = args.sentences
        self.iterations = args.iterations

        self.dictionary = np.ones(shape=[eng_size, cz_size])
        self.dictionary /= cz_size
    
    def run(self, corpus):
        """
            This method implements IBM1 model and EM algorithm
        """
        for iteration in range(1, self.iterations + 1):
            print(f"current iteration: {iteration}")
            count = np.zeros(shape=[len(corpus.unique_eng), len(corpus.unique_cz)])
            total = np.zeros(len(corpus.unique_cz))
            total_s = {}

            with open(corpus.file_location, "r", encoding="utf8") as file:
                line_count = 0
                for line in file:

                    if self.lower: 
                        line = line.lower()

                    columns = line.split('\t')
                    eng_sentence = columns[0].split() 
                    cz_sentence  = columns[1].split() 

                    for eng_word in eng_sentence:
                        if is_punctuation(self.punctuation, eng_word):
                            continue

                        total_s[eng_word] = 0

                        for cz_word in cz_sentence:
                            if is_punctuation(self.punctuation, cz_word):
                                continue

                            eng_index = corpus.unique_eng[eng_word]
                            cz_index  = corpus.unique_cz[cz_word]
                        
                            total_s[eng_word] += self.dictionary[eng_index][cz_index]
                    
                    word_cache = set()

                    for eng_word in eng_sentence:
                        if is_punctuation(self.punctuation, eng_word):
                                continue

                        for cz_word in cz_sentence:
                            if is_punctuation(self.punctuation, cz_word):
                                continue

                            eng_index = corpus.unique_eng[eng_word]
                            cz_index  = corpus.unique_cz[cz_word]

                            count[eng_index][cz_index] += self.dictionary[eng_index][cz_index] / total_s[eng_word]

                            word_cache.add((eng_index, cz_index))

                            total[cz_index] += self.dictionary[eng_index][cz_index] / total_s[eng_word]

                    for (eng, cz) in word_cache:
                        self.dictionary[eng][cz] = count[eng][cz] / total[cz]
                
                    line_count += 1
                    if line_count == self.sentences:
                        break

def is_punctuation(allowed, word):
    return not allowed and word in string.punctuation
        
def get_best_translations(data: Corpus, eng_col, top): 
    """
        Returns a list of words which are the top translations 
        based on the translation table
    """
    n_best_results = np.argpartition(eng_col, -top)[-top:]
    n_best_results.sort()

    word_list = []
    for i in n_best_results:
        eng_word = data.reverse_unique_eng[i]
        word_list.append(eng_word)

    return word_list
    
def report_result(data: Corpus, model_one: WordAlignment, top_n): 
    """
        This method writes the results to the output file. It calls 
        the function that gets best n translations and writes them 
        to the file
    """
    file_name = "output/results.txt"
    with open(file_name, "w", encoding="utf8") as output_file:
        for cz_word in data.unique_cz:
            output_file.write(cz_word + ':\t')
            index = data.unique_cz[cz_word]
            eng_column = model_one.dictionary[:, index]

            top_n_result = get_best_translations(data, eng_column, top_n)

            output_file.write(" ".join(top_n_result) + "\n")

def main(args: argparse.Namespace): 
    data = Corpus(args)

    model_one = WordAlignment(len(data.unique_eng), len(data.unique_cz), args)
    model_one.run(data)

    report_result(data, model_one, args.top)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations to perform EM algorithm")
    parser.add_argument("--sentences", type=int, default=2000, help="Number of sentences from the eng-cz corpus")
    parser.add_argument("--top", type=int, default=3, help="Returns best n translations of the current CZ word")
    parser.add_argument("--punctuation_allowed", action="store_true", default=True, help="Remove punctuations in the preprocessing")
    parser.add_argument("--to_lower", action="store_true", default=False, help="Lowers the casing of words in the preprocessing")

    args = parser.parse_args([] if "__file__" not in globals() else None)
    
    main(args)