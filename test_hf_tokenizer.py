from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import Regex, Tokenizer, models, decoders, trainers, processors, pre_tokenizers



def test_hf_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]", byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(
                    "\t|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    # noqa: E501
                ),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)

    trainer = trainers.BpeTrainer(
        vocab_size=5000,
        special_tokens=["<|alex|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        continuing_subword_prefix="",
        end_of_word_suffix="",
        show_progress=True
    )

    # files = ["./tests/fixtures/corpus.en"]
    files = ["./data/TinyStoriesV2-GPT4-train.txt"]
    tokenizer.train(files, trainer)
    tokenizer.save("testing-hf.json")