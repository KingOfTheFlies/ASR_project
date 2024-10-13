import re
from string import ascii_lowercase

from collections import defaultdict
import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = "^"

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        text_to_decode = []
        prev_ind = None
        blank_index = self.char2ind[self.EMPTY_TOK]

        for ind in inds:
            if ind != prev_ind and ind != blank_index:
                text_to_decode.append(ind)
            prev_ind = ind

        return self.decode(text_to_decode)


    def expand_and_merge_path(self, beams, current_probs):
        next_beams = defaultdict(float)
        blank_token = self.EMPTY_TOK
        vocab_size = len(self.vocab)

        for prefix, last_char, score in beams:
            for idx in range(vocab_size):
                char = self.ind2char[idx]
                prob = current_probs[idx]
                new_score = score * prob

                if char == blank_token:
                    new_prefix = prefix
                elif char == last_char:
                    new_prefix = prefix
                else:
                    new_prefix = prefix + char

                key = (new_prefix, char)
                next_beams[key] += new_score

        next_beams_list = [(prefix, last_char, score) for (prefix, last_char), score in next_beams.items()]
        return next_beams_list
        
    def truncate_paths(self, beams, beam_size):
        sorted_beams = sorted(beams, key=lambda x: x[2], reverse=True)
        truncated_beams = sorted_beams[:beam_size]
        return truncated_beams
    
    def ctc_beam_search_decode(self, probs, beam_size=10):
        """
        Выполняет CTC beam search декодирование.

        Args:
            probs (torch.Tensor или np.ndarray): Массив вероятностей размерности (T, vocab_size).
            beam_size (int): Максимальное количество beam для сохранения на каждом шаге.

        Returns:
            List[Dict[str, float]]: Список гипотез с их вероятностями.
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        time_steps, vocab_size = probs.shape
        blank_token = self.EMPTY_TOK

        beams = [('', blank_token, 1.0)]

        for t in range(time_steps):
            current_probs = probs[t]
            next_beams = self.expand_and_merge_path(beams, current_probs)
            beams = self.truncate_paths(next_beams, beam_size)

        final_results = []
        for prefix, _, score in beams:
            final_results.append({'text': prefix, 'prob': score})

        final_results.sort(key=lambda x: x['prob'], reverse=True)

        return final_results


    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
