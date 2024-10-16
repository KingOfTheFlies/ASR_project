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

    def __init__(self, use_bpe, bpe_vocab_size, beam_size, 
                 tokenizer_model_dir, train_text_path, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        self.use_bpe = use_bpe
        self.tokenizer = None
        self.beam_size = beam_size

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
                lowercase=True
            )

            self.vocab = self.tokenizer.get_vocab()

        else:
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
            self.ind2char = dict(enumerate(self.vocab))
            self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ind2token(self, ind):
        if self.use_bpe:
            return self.tokenizer.id_to_token(ind) 
        else:
            return self.ind2char[ind]
            
    def token2ind(self, token):
        if self.use_bpe:
            return self.tokenizer.token_to_id(token)
        else:
            return self.char2ind[token]
            

    def text2ind(self, text):
        if self.use_bpe:
            return torch.Tensor(self.tokenizer.encode(text).ids)
        else:
            return torch.Tensor([self.token2ind(token) for token in text]).unsqueeze(0)
            

    def ind2text(self, inds):
        if self.use_bpe:
            return self.tokenizer.decode(inds)
        else:            
            return "".join([self.ind2token(int(ind)) for ind in inds]).strip()

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.text2ind(char) for char in text]).unsqueeze(0)
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
        return self.ind2text(inds)

    def ctc_decode(self, inds) -> str:
        text_to_decode = []
        prev_ind = None
        blank_index = self.token2ind(self.EMPTY_TOK)

        for ind in inds:
            if ind != prev_ind and ind != blank_index:
                text_to_decode.append(ind)
            prev_ind = ind

        return self.decode(text_to_decode)

    def ctc_beam_search_decode(self, probs):
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        time_steps, vocab_size = probs.shape
        blank_token = self.EMPTY_TOK

        beams = [('', blank_token, 1.0)]

        for t in range(time_steps):
            current_probs = probs[t]
            next_beams = self.expand_and_merge_path(beams, current_probs)
            beams = self.truncate_paths(next_beams, self.beam_size)
        final_results = []
        for prefix, _, score in beams:
            final_results.append({'text': prefix, 'prob': score})
        final_results.sort(key=lambda x: x['prob'], reverse=True)
        return final_results

    def expand_and_merge_path(self, beams, current_probs):
        next_beams = defaultdict(float)
        blank_token = self.EMPTY_TOK
        vocab_size = len(self.vocab)

        for prefix, last_char, score in beams:
            for idx in range(vocab_size):
                char = self.ind2token(idx)
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

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
