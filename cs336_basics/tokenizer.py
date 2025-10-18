import os
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from tests.common import gpt2_bytes_to_unicode

import regex as re

PAT_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")


def pre_tokenize(text_chunk: str, special_tokens: list[str], skip_specials=True) -> Iterator[bytes]:
    if len(special_tokens) == 0:
        for m in PAT_RE.finditer(text_chunk):
            yield m.group().encode()
        return

    escaped_specials = sorted((re.escape(t) for t in special_tokens), key=len, reverse=True)
    special_re_txt = "|".join(escaped_specials)
    if not skip_specials:
        special_re_txt = f"({special_re_txt})"
    special_token_split_re = re.compile(special_re_txt)
    for word in special_token_split_re.split(text_chunk):
        if word in special_tokens:
            yield word.encode()
            continue

        for m in PAT_RE.finditer(word):
            yield m.group().encode()


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = list(set(special_tokens + ["<|endoftext|>"]))  # ensure endoftext is present

    min_vocab_size = 256 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(f"vocab_size must be at least 256+specials={min_vocab_size}, got {vocab_size}")

    # base vocab
    id2tok = [s.encode() for s in special_tokens] + [bytes([i]) for i in range(256)]
    tok2id = {tok: idx for idx, tok in enumerate(id2tok)}

    # TODO: parallelize once working
    # upd: this is not a bottleneck comparing to merges, fix that first
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    # num_processes = os.cpu_count() or 4
    # with open(input_path, "rb") as f:
    #     boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # count pre-tokens
    word_counts = Counter(pre_tokenize(text, special_tokens))

    words_tokenized = []
    counts = []

    for word, count in word_counts.items():
        words_tokenized.append([tok2id[bytes([b])] for b in word])
        counts.append(count)

    merges: list[tuple[bytes, bytes]] = []

    # count pair stats once and update in place during merge
    pair_stats = Counter()
    for chunk, count in zip(words_tokenized, counts):
        for j in range(len(chunk) - 1):
            pair = (chunk[j], chunk[j + 1])
            pair_stats[pair] += count

    while len(id2tok) < vocab_size:
        # TODO: on the parallel - collect Δs and merge
        # TODO: use some sorted queue datastructure instead of finding max every time
        max_count = max(pair_stats.values())
        if max_count < 0:
            break
        max_pairs = [pair for pair, count in pair_stats.items() if count == max_count]
        # Among ties, pick the lexicographically greatest
        top_pair = max(
            max_pairs,
            key=lambda pair: (id2tok[pair[0]], id2tok[pair[1]]),
        )

        # merge
        pair_bytes = (id2tok[top_pair[0]], id2tok[top_pair[1]])
        merges.append(pair_bytes)
        new_token_idx = len(id2tok)
        id2tok.append(pair_bytes[0] + pair_bytes[1])
        del pair_stats[top_pair]  # clear stat for merged token

        # print(f"merging {new_token_idx}: pair:{pair_bytes}->{pair_bytes[0] + pair_bytes[1]} count:{max_count}")

        for index, chunk in enumerate(words_tokenized):
            idx_to_delete = []
            j = 0
            while j < len(chunk):
                if j < len(chunk) - 1 and (chunk[j], chunk[j + 1]) == top_pair:
                    # update stats
                    c = counts[index]
                    if j < len(chunk) - 2:
                        to_the_right_token = chunk[j + 2]
                        pair_stats[(chunk[j + 1], to_the_right_token)] -= c
                        pair_stats[(new_token_idx, to_the_right_token)] += c
                    if j > 0:
                        to_the_left_token = chunk[j - 1]
                        pair_stats[(to_the_left_token, chunk[j])] -= c
                        pair_stats[(to_the_left_token, new_token_idx)] += c
                    # merge matched
                    chunk[j] = new_token_idx
                    idx_to_delete.append(j + 1)
                    j += 1
                j += 1
            if len(idx_to_delete) > 0:
                words_tokenized[index] = [item for i, item in enumerate(chunk) if i not in idx_to_delete]

    return {i: t for i, t in enumerate(id2tok)}, list(merges)


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.special_tokens = special_tokens or []
        self.id2tok = vocab

        # we expect specials to be in vocab
        # self.id2tok.update({len(vocab) + i: st.encode() for i, st in enumerate(self.special_tokens)})

        self.tok2id = {b: i for i, b in vocab.items()}
        self.merges = merges

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        # GPT-2 bytes to unicode mapping (for decoding)
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        # Load vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)

        # Convert vocab from GPT-2 unicode format to bytes
        vocab = {}
        for idx_str, gpt2_str in vocab_json.items():
            idx = int(idx_str)
            byte_list = [gpt2_byte_decoder[char] for char in gpt2_str]
            vocab[idx] = bytes(byte_list)

        # Load merges
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or len(line.split()) != 2:
                    continue
                a_str, b_str = line.split()
                a_bytes = bytes([gpt2_byte_decoder[char] for char in a_str])
                b_bytes = bytes([gpt2_byte_decoder[char] for char in b_str])
                merges.append((a_bytes, b_bytes))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def _to_token_words(self, s: str) -> Iterator[list[int]]:
        for word in pre_tokenize(s, special_tokens=self.special_tokens, skip_specials=False):
            if word in self.tok2id:
                yield [self.tok2id[word]]  # single token
                continue
            yield [self.tok2id[bytes([b])] for b in word]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            rank_not_found_max = len(self.tok2id) + 1000

            for word_tokens in self._to_token_words(text_chunk):  # PERF: parallelizable
                word_tokens_mut = word_tokens
                scan_start = 0  # allows to start from first found pair position instead of zero. kinda optimization

                # keep iter inside word until no pair is mergeable
                while True:
                    min_rank_candidate_token = rank_not_found_max
                    merge_position = -1
                    current_start = scan_start
                    for i, (t1, t2) in enumerate(
                        zip(word_tokens_mut[current_start:-1], word_tokens_mut[current_start + 1 :])
                    ):
                        maybe_pair = self.id2tok[t1] + self.id2tok[t2]
                        rank = self.tok2id.get(maybe_pair, None)
                        if rank is None:
                            if merge_position == -1:
                                scan_start = max(
                                    0,
                                    current_start + i,
                                )  # updating start to before first pair, since it might be not the lowest rank
                            continue
                        # print(f"t1={t1} t2={t2}, pair={maybe_pair} rank={rank}")
                        if rank < min_rank_candidate_token:
                            min_rank_candidate_token = rank
                            merge_position = current_start + i

                    if merge_position == -1:
                        break
                    # print(f"before merge {scan_start=} {merge_position=} {word_tokens_mut=}")
                    word_tokens_mut = (
                        word_tokens_mut[:merge_position]
                        + [min_rank_candidate_token]
                        + word_tokens_mut[merge_position + 2 :]
                    )
                    # print(f"after merge {scan_start=} {word_tokens_mut=}")

                yield from word_tokens_mut

    def decode(self, ids: list[int]) -> str:
        raw_bytes = b"".join([self.id2tok[tok_id] for tok_id in ids])
        return bytes.decode(raw_bytes, errors="replace")


def test_encoding():
    # vocab :  dict[int, bytes]
    t = Tokenizer(vocab={0: b"<|alex|>", 1: b"a", 2: b"b", 3: b" ", 4: b"ab"}, merges=[], special_tokens=["<|alex|>"])
    tokens = t.encode("aaabab ab")
    print(f"{tokens=}")
    assert tokens == [1, 1, 4, 4, 3, 4]


def _test_bpe_encoding():
    # vocab :  dict[int, bytes]
    vocab = {
        0: b" ",
        1: b"a",
        2: b"c",
        3: b"e",
        4: b"h",
        5: b"t",
        6: b"th",
        7: b" c",
        8: b" a",
        9: b"the",
        10: b" at",
    }
    t = Tokenizer(
        vocab=vocab,
        merges=[],
        special_tokens=[],
    )
    text = "the cat ate"
    tokens = t.encode(text)
    print(f"{tokens=}")

    def debug_tokens(tokens: list[int]):
        return [(vocab[i], i) for i in tokens]

    assert debug_tokens(tokens) == debug_tokens([9, 7, 1, 5, 10, 3])
    assert t.decode(tokens) == text
