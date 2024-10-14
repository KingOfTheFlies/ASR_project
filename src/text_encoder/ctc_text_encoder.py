import re
from string import ascii_lowercase

from collections import defaultdict
import torch
import os

import tempfile
from tokenizers import ByteLevelBPETokenizer
import json

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = "^"

    def __init__(self, use_bpe, bpe_vocab_size, tokenizer_model_dir, train_text_path, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        self.use_bpe = use_bpe
        self.tokenizer = None

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet

        if self.use_bpe:
            vocab_file = os.path.join(tokenizer_model_dir, 'bpe_tokenizer-vocab.json')
            merges_file = os.path.join(tokenizer_model_dir, 'bpe_tokenizer-merges.txt')

            if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
                with open(train_text_path) as f:
                    data = json.load(f)
                    corpus_data = '\n'.join([txt['text'] for txt in data])


                with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp_corpus_file:
                    tmp_corpus_file.write(corpus_data)
                    tmp_corpus_file_name = tmp_corpus_file.name

                if not os.path.exists(tokenizer_model_dir):
                    os.makedirs(tokenizer_model_dir)

                tokenizer = ByteLevelBPETokenizer()
                tokenizer.train(files=[tmp_corpus_file_name], vocab_size=bpe_vocab_size, special_tokens=[self.EMPTY_TOK])

                tokenizer.save_model(tokenizer_model_dir, 'bpe_tokenizer')

                os.remove(tmp_corpus_file_name)
            
            
            self.tokenizer = ByteLevelBPETokenizer(
                vocab_file,
                merges_file,
                add_prefix_space=False,
                lowercase=True
            )

            tokenizer_vocab = self.tokenizer.get_vocab()
            self.char2ind = {token: index + 1 for token, index in tokenizer_vocab.items()}
            
            self.char2ind[self.EMPTY_TOK] = 0
            self.ind2char = {index: token for token, index in self.char2ind.items()}
            self.vocab = [self.EMPTY_TOK] + list(tokenizer_vocab.keys())

        else:
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
        if self.use_bpe and self.tokenizer is not None:
            encoded = self.tokenizer.encode(text)
            token_ids = encoded.ids
            token_ids = [tid + 1 for tid in token_ids]
            return torch.tensor(token_ids).unsqueeze(0)

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
        if self.use_bpe and self.tokenizer is not None:
            token_ids = [int(ind) - 1 for ind in inds if int(ind) != 0]
            tokens = [self.tokenizer.id_to_token(tid) for tid in token_ids]
            text = self.tokenizer.decoder.decode(tokens)
            return text
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


    def expand_end_merge_path(self, beams, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in beams.items():
                if last_char == cur_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp
        
    def truncate_paths(beams, beam_size):
        return dict(sorted(list(beams.items()), key=lambda x: -x[1])[:beam_size])
    
    def ctc_beam_search_decode(self, probs, beam_size):
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        beams = { ("", self.EMPTY_TOK): 1. }

        for prob in probs:
            beams = self.expand_end_merge_path(beams, prob)
            beams = self.truncate_paths(beams, beam_size)

        beams = [
            {"text": prefix, "prob": proba.item()}  \
            for (prefix, _), proba in sorted(beams.items(), key=lambda x: -x[1])
        ]
        return beams


    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
