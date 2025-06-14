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

    num_merges = vocab_size - min_vocab_size

    # base vocab
    id2tok = {i: s.encode() for i, s in enumerate(special_tokens)}
    id2tok |= {len(special_tokens) + i: bytes([i]) for i in range(256)}
    tok2id = {tok: idx for idx, tok in id2tok.items()}
    initial_vocab_size = len(id2tok)  # 256 + len(special_tokens)

    # iterator over tokenized words
    escaped_specials = sorted((re.escape(t) for t in special_tokens), key=len, reverse=True)
    special_token_split_re = re.compile(f"({'|'.join(escaped_specials)})")

    def tokenize_chunk_iter(text: str) -> Iterator[list[int]]:
        for word in special_token_split_re.split(text):
            if not word:  # empty split piece → ignore
                continue
            if word in special_tokens:
                yield [tok2id[word.encode("utf-8")]]
            else:
                for m in PAT_RE.finditer(word):
                    word_bytes = m.group().encode("utf-8")
                    yield [tok2id[bytes([b])] for b in word_bytes]

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
    token_chunks: list[list[int]] = list(tokenize_chunk_iter(text))

    merges: list[tuple[bytes, bytes]] = []

    for i in range(num_merges):
        # TODO: count once and update in-place during merge 0 -1 (old) -1 0 => 0 1 1 0
        # TODO: on the parallel - collect Δs and merge
        stats = Counter()
        for chunk in token_chunks:
            for j in range(len(chunk) - 1):
                pair = (chunk[j], chunk[j + 1])
                stats[pair] += 1
        max_count = max(stats.values())
        # Get all pairs with max count
        max_pairs = [pair for pair, count in stats.items() if count == max_count]
        # Among ties, pick the lexicographically greatest
        top_pair = max(
            max_pairs,
            key=lambda pair: (id2tok[pair[0]], id2tok[pair[1]]),
        )

        # merge
        new_token_idx = initial_vocab_size + i
        pair_bytes = (id2tok[top_pair[0]], id2tok[top_pair[1]])
        merges.append(pair_bytes)
        id2tok[new_token_idx] = pair_bytes[0] + pair_bytes[1]
        print(f"merging {new_token_idx}: pair:{pair_bytes}->{pair_bytes[0] + pair_bytes[1]} count:{max_count}")

        new_token_chunks = []
        for chunk in token_chunks:
            new_chunk = []
            j = 0
            while j < len(chunk):
                if j < len(chunk) - 1 and (chunk[j], chunk[j + 1]) == top_pair:
                    new_chunk.append(new_token_idx)
                    j += 2
                else:
                    new_chunk.append(chunk[j])
                    j += 1
            new_token_chunks.append(new_chunk)

        token_chunks = new_token_chunks

    return id2tok, list(merges)
