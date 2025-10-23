"""Script to tokenize the TinyStories training dataset and save to numpy array."""
import numpy as np
import pathlib
import time
from cs336_basics.tokenizer import Tokenizer


class ProgressTracker:
    """Track and report tokenization progress."""

    def __init__(self, input_file_size: int, report_interval: int = 1_000_000):
        self.input_file_size = input_file_size
        self.input_file_size_mb = input_file_size / (1024**2)
        self.report_interval = report_interval
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.last_report_tokens = 0

    def report(self, token_count: int, bytes_read: int):
        """Print progress report."""
        current_time = time.time()
        elapsed_since_report = current_time - self.last_report_time
        tokens_since_report = token_count - self.last_report_tokens

        # Calculate speeds
        tokens_per_sec = tokens_since_report / elapsed_since_report
        mb_read = bytes_read / (1024**2)
        mb_per_sec = mb_read / (current_time - self.start_time)
        percent_complete = (bytes_read / self.input_file_size) * 100

        print(f"Processed {token_count:,} tokens | "
              f"{mb_read:.1f}/{self.input_file_size_mb:.1f} MB ({percent_complete:.1f}%) | "
              f"Speed: {tokens_per_sec:,.0f} tok/s, {mb_per_sec:.2f} MB/s")

        self.last_report_time = current_time
        self.last_report_tokens = token_count

    def final_report(self, token_count: int):
        """Print final summary."""
        total_time = time.time() - self.start_time
        avg_tokens_per_sec = token_count / total_time
        avg_mb_per_sec = self.input_file_size_mb / total_time

        print(f"\nTotal time: {total_time:.2f} seconds")
        print(f"Average speed: {avg_tokens_per_sec:,.0f} tokens/s, {avg_mb_per_sec:.2f} MB/s")


def tokenize_and_save(
    input_path: str,
    output_path: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str] | None = None,
    chunk_size: int = 100000,  # Buffer size for writing tokens
):
    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    # Setup progress tracking
    input_file_size = pathlib.Path(input_path).stat().st_size
    progress = ProgressTracker(input_file_size)
    print(f"Input file size: {progress.input_file_size_mb:.2f} MB")
    print(f"Tokenizing {input_path} and streaming to {output_path}...")

    token_count = 0
    chunk = []

    with open(input_path, 'r', encoding='utf-8') as infile:
        with open(output_path, 'wb') as outfile:
            for token_id in tokenizer.encode_iterable(infile):
                chunk.append(token_id)
                token_count += 1

                # Write chunk to disk when buffer is full
                if len(chunk) >= chunk_size:
                    np.array(chunk, dtype=np.uint16).tofile(outfile)
                    chunk = []

                    # Progress report
                    if token_count % progress.report_interval == 0:
                        progress.report(token_count, infile.buffer.tell())

            # Write remaining tokens
            if chunk:
                np.array(chunk, dtype=np.uint16).tofile(outfile)

    # Final summary
    print(f"\nDone! Wrote {token_count:,} tokens to {output_path}")
    progress.final_report(token_count)
    file_size_mb = pathlib.Path(output_path).stat().st_size / (1024**2)
    print(f"Output file size: {file_size_mb:.2f} MB")

    return token_count


def load_tokenized_dataset(bin_path: str) -> np.ndarray:
    print(f"Loading tokenized dataset from {bin_path} (memory-mapped)...")
    token_array = np.memmap(bin_path, dtype=np.uint16, mode='r')
    print(f"Loaded {len(token_array):,} tokens (memory-mapped)")
    return token_array


if __name__ == "__main__":
    # Paths
    base_dir = pathlib.Path(__file__).parent.parent.parent # I know..
    input_file = base_dir / "data" / "TinyStoriesV2-GPT4-train.txt"
    vocab_file = base_dir / "tinystories_vocab.json"
    merges_file = base_dir / "tinystories_merges.txt"
    output_file = base_dir / "tinystories_train_tokenized.bin"

    # Tokenize and save
    tokenize_and_save(
        input_path=str(input_file),
        output_path=str(output_file),
        vocab_path=str(vocab_file),
        merges_path=str(merges_file),
        special_tokens=["<|endoftext|>"],
    )

    # Test loading
    print("\nTesting load...")
    tokens = load_tokenized_dataset(str(output_file))
    print(f"First 20 tokens: {tokens[:20]}")
    print(f"Last 20 tokens: {tokens[-20:]}")
