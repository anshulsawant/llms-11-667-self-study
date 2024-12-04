import datasets as ds

from tokenizer.bpe import UnicodeBPETokenizer

just_some_text = "\n".join(
    row["text"]
    for row in ds.load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )
    .shuffle(seed=42)
    .take(500)
)

tokenizer = UnicodeBPETokenizer.from_data(just_some_text, 256)

tokenizer.save("data/english-tokenizer.json")
