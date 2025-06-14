import os
import regex as re
from collections import Counter
from collections.abc import Iterator

PAT_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = list(set(special_tokens + ["<|endoftext|>"]))  # ensure endoftext is present

    min_vocab_size = 256 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(f"vocab_size must be at least {min_vocab_size}, got {vocab_size}")

    # base vocab
    id2tok = [s.encode() for s in special_tokens] + [bytes([i]) for i in range(256)]
    tok2id = {tok: idx for idx, tok in enumerate(id2tok)}

    # iterator over tokenized words
    escaped_specials = sorted((re.escape(t) for t in special_tokens), key=len, reverse=True)
    special_token_split_re = re.compile("|".join(escaped_specials))

    def pre_tokenize(text_chunk: str) -> Iterator[bytes]:
        for word in special_token_split_re.split(text_chunk):
            for m in PAT_RE.finditer(word):
                yield m.group().encode()

    # TODO: parallelize once working
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    # num_processes = os.cpu_count() or 4
    # with open(input_path, "rb") as f:
    #     boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #

    # count pre-tokens
    word_counts = Counter(pre_tokenize(text))

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
                    to_the_left_token = chunk[j - 1] if j > 0 else -1
                    to_the_right_token = chunk[j + 2] if j < len(chunk) - 2 else -1
                    if to_the_right_token != -1:
                        pair_stats[(chunk[j + 1], to_the_right_token)] -= c
                        pair_stats[(new_token_idx, to_the_right_token)] += c
                    if to_the_left_token != - 1:
                        pair_stats[(to_the_left_token, chunk[j])] -= c
                        pair_stats[(to_the_left_token, new_token_idx)] += c

                    chunk[j] = new_token_idx
                    idx_to_delete.append(j+1)
                    j += 1
                j += 1
            if len(idx_to_delete) > 0:
                words_tokenized[index] = [item for i, item in enumerate(chunk) if i not in idx_to_delete]


    return {i: t for i, t in enumerate(id2tok)}, list(merges)
