import random
import re
import string

import numpy as np


class EnigmaForDummies:
    def __init__(self) -> None:
        self.emp_freq_prepared = False
        self.regex_pattern = re.compile("[^a-z ]")
        self.alphabet = " " + string.ascii_lowercase
        self.char_index_mapping = dict(
            zip(list(self.alphabet), range(len(self.alphabet)))
        )

    def encrypt_or_decrypt(self, text: str, mapping: dict) -> str:
        """Transform text using mapping dictionary

        Args:
            text (str): text_to_transform
            mapping (dict): mapping dictionary for symbols transformation

        Returns:
            str: transformed text
        """
        return "".join([mapping[s] for s in text])

    def prepare_empirical_freq_normalized(self, path_to_file: str) -> None:
        """Prepare dictionary of normalized empirical frequences from provided corpus

        Args:
            path_to_file (str): path to corpus text file
        """
        # Dictionary to store bigram pairs counts
        char_bigram_counts = {}

        with open(path_to_file, encoding="utf-8") as f:
            for line in f:
                # pre-process line
                clean_line = np.array(list(self.regex_pattern.sub("", line.lower())))

                # create symbols transition array
                transitions = clean_line.repeat(2)[1:-1].reshape(-1, 2)

                # count every symbols transition met in line
                for i, j in transitions:
                    char_bigram_counts[(i, j)] = (
                        char_bigram_counts.setdefault((i, j), 0) + 1
                    )

        # create letters encoder
        letters = [" "] + list(string.ascii_lowercase)
        unigram_to_index = dict(zip(letters, range(len(letters))))

        # Create transition matrix
        n = len(unigram_to_index)
        transition_matrix = np.ones((n, n)) + 1

        # fill in transition matrix for each pair
        for s_pair in char_bigram_counts.keys():
            transition_matrix[unigram_to_index[s_pair[0]]][
                unigram_to_index[s_pair[1]]
            ] = char_bigram_counts[s_pair]

        # normalize matrix rows
        row_sums = transition_matrix.sum(axis=1)
        self.empirical_frequences = transition_matrix / row_sums[:, np.newaxis]
        self.emp_freq_prepared = True

    def encrypt_text(self, text: str) -> str:
        """Perform encryption of text by randomly mapping each letter to other letter in alphabet

        Args:
            text (str): input text to encrypt

        Returns:
            str: encrypted text
        """
        text_cleaned = self.regex_pattern.sub("", text.lower())
        shuffled_alphabet = list(self.alphabet)
        random.shuffle(shuffled_alphabet)
        random_cifer = dict(zip(list(self.alphabet), shuffled_alphabet))
        encrypted_text = self.encrypt_or_decrypt(text_cleaned, random_cifer)
        return encrypted_text

    def score_cipher(self, cipher: dict, encrypted: str) -> float:
        """Score current cipher given encrypted text. The higher score the better

        Args:
            cipher (dict): proposed cipher to score
            encrypted (str): encrypted text

        Returns:
            (float or str): cipher score or decrypted sample
        """
        decrypted = self.encrypt_or_decrypt(
            encrypted, {v: k for k, v in cipher.items()}
        )

        score = 0
        for i in range(len(decrypted) - 1):
            first_symbol = self.char_index_mapping[decrypted[i]]
            second_symbol = self.char_index_mapping[decrypted[i + 1]]
            score += np.log(self.empirical_frequences[first_symbol][second_symbol])

        return score

    def process_decryption(self, encrypted: str, iters: int = 5000, verbose=500) -> str:
        """Process text decryption using MCMC algorithm with random cipher permutations

        Args:
            encrypted (str): encrypted text
            iters (int, optional): _description_. Defaults to 5000.
            verbose (int, optional): _description_. Defaults to 500.

        Returns:
            str: _description_
        """
        assert self.emp_freq_prepared, "Prepare empirical frequences first."
        # Initialize with a random mapping
        shuffled_letters = list(self.alphabet)
        random.shuffle(shuffled_letters)
        current_cifer = dict(zip(shuffled_letters, list(self.alphabet)))
        current_score = self.score_cipher(current_cifer, encrypted)

        best_cifer, best_score = current_cifer.copy(), current_score
        for i in range(0, iters):
            # Create proposal from f by random transposition of 2 letters
            r1, r2 = np.random.choice(list(shuffled_letters), 2, replace=True)
            new_cifer = current_cifer.copy()
            new_cifer[r1] = current_cifer[r2]
            new_cifer[r2] = current_cifer[r1]
            new_score = self.score_cipher(new_cifer, encrypted)

            # Decide to accept new proposal
            if new_score > current_score or random.uniform(0, 1) < np.exp(
                new_score - current_score
            ):
                current_cifer = new_cifer.copy()
                current_score = new_score

            if new_score > best_score:
                best_score = new_score
                best_cifer = new_cifer.copy()

            # Print out progress
            if i % 500 == 0 and verbose:
                assert (
                    type(verbose) is int and verbose > 0
                ), "Select verbose=False or pass positive integer"

                if i % verbose == 0:
                    best_attempt_smpl = self.encrypt_or_decrypt(
                        encrypted, {v: k for k, v in best_cifer.items()}
                    )[:100]

                    print(i, ":\t", best_attempt_smpl)

        # Save best mapping
        cipher_alphabet = "".join([k for k in sorted(best_cifer.keys())])
        plaintext_alphabet = "".join([best_cifer[k] for k in cipher_alphabet])
        mapping = dict(zip(plaintext_alphabet, cipher_alphabet))

        decrypted_text = self.encrypt_or_decrypt(encrypted, mapping)

        return decrypted_text


if __name__ == "__main__":
    enigma = EnigmaForDummies()
    enigma.prepare_empirical_freq_normalized("war_and_peace.txt")

    plain_text = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'
So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before see a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.
In another moment down went Alice after it, never once considering how in the world she was to get out again.
The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.
"""
    encrypted = enigma.encrypt_text(plain_text)
    print("Encrypted (first 500):\n", encrypted[:500], end="\n\n")

    decrypted = enigma.process_decryption(encrypted)
    print("\nDecrypted (first 500):\n", decrypted[:500])
