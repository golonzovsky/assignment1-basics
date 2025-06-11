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
    min_vocab_size = 256 + 1 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(f"vocab_size must be at least 257+len(special_tokens)={min_vocab_size}, got {vocab_size}")

    num_merges = vocab_size - min_vocab_size

    id2tok = {i: bytes([i]) for i in range(256)}
    id2tok |= {256 + i: s.encode() for i, s in enumerate(special_tokens)}

    # print("vocab_initial", id2tok)
    initial_vocab_size = len(id2tok)

    tok2id = {tok: idx for idx, tok in id2tok.items()}

    # TODO: parallelize once working
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    escaped_specials = sorted((re.escape(t) for t in special_tokens), key=len, reverse=True)
    special_token_split_re = re.compile(f"({'|'.join(escaped_specials)})")

    def tokenize_chunk_iter(text: str) -> Iterator[bytes]:
        for chunk in special_token_split_re.split(text):
            if not chunk:  # empty split piece → ignore
                continue
            if chunk in special_tokens:
                yield chunk.encode()
            else:
                for m in PAT_RE.finditer(chunk):
                    yield m.group().encode("utf-8")

    token_chunks: list[bytes] = list(tokenize_chunk_iter(text))

    merges: list[tuple[bytes, bytes]] = []

    for i in range(num_merges):
        # TODO count once and update during merge
        stats = Counter()
        for chunk in token_chunks:
            for j in range(len(chunk) - 1):
                pair = (chunk[j], chunk[j + 1])
                stats[pair] += 1
        # top_pair, top_pair_count = stats.most_common(1)[0]
        top_pair, top_pair_count = max(stats.items(), key=lambda item: (item[1], item[0]))

        # merge
        new_token_idx = initial_vocab_size + i
        pair_bytes = (id2tok[top_pair[0]], id2tok[top_pair[1]])
        merges.append(pair_bytes)
        id2tok[new_token_idx] = pair_bytes[0] + pair_bytes[1]

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

