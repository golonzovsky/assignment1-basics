import pathlib
import json

from cs336_basics.tokenizer import train_bpe

from tests.common import gpt2_bytes_to_unicode

def bytes_to_gpt2_string(data: bytes) -> str:
    byte_to_unicode = gpt2_bytes_to_unicode()
    return ''.join(byte_to_unicode[b] for b in data)

def save_vocab(vocab: dict[int, bytes], filename: str) -> None:
    json_vocab = {str(k): bytes_to_gpt2_string(v) for k, v in vocab.items()}
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_vocab, f, indent=2, ensure_ascii=False)


def save_merges(merges: list[tuple[bytes, bytes]], filename: str) -> None:
    with open(filename, 'w', encoding='utf-8') as f:
        for a, b in merges:
            f.write(f"{bytes_to_gpt2_string(a)} {bytes_to_gpt2_string(b)}\n")

def train_bpe_tiny_stories():
    # input_path = pathlib.Path(__file__).resolve().parent / "data" / "TinyStoriesV2-GPT4-valid.txt" # 29.85 sec
    input_path =  pathlib.Path(__file__).resolve().parent.parent / "data" / "TinyStoriesV2-GPT4-train.txt" # 6m
    # input_path =  pathlib.Path(__file__).resolve() / "fixtures" / "tinystories_sample_5M.txt"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    save_vocab(vocab, "./tinystories_vocab.json")
    save_merges(merges, "./tinystories_merges.txt")


if __name__ == "__main__":
    train_bpe_tiny_stories()