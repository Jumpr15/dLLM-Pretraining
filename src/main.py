from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader

# load tokenizer and add special pad and mask tokens if missing
def prepare_tokenizer(hf_tokenizer, default_pad="<|pad|>", default_mask="<|mask|>"):
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    if tokenizer.pad_token is None:
        print("tokenizer has no pad_token: adding default pad token")
        tokenizer.add_special_tokens({
            "pad_token": default_pad
        })
    if tokenizer.mask_token is None:
        print("tokenizer has no mask_token: adding default mask token")
        tokenizer.add_special_tokens({
            "mask_token": default_mask
        })
    return tokenizer 

# loads IterableDataset from hf for streaming
def prepare_dataset(hf_dataset, tokenizer, seq_len, tokenize_column="text", remove_columns=[], split="train"):
    dataset = load_dataset(
        hf_dataset,
        split=split,
        streaming=True
    )
    dataset = dataset.remove_columns(remove_columns)
    
    def tokenize_fn(example):
        return tokenizer(
            example[tokenize_column],
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len
        )

    
    dataset = dataset.map(tokenize_fn)

def prepare_dataloader():
    tokenizer = prepare_tokenizer(
        "HuggingFaceTB/SmolLM-1.7B"
    )
    dataset = prepare_dataset(
        "OptimalScale/ClimbMix",
        tokenizer,
        64,
        tokenize_column="text",
        remove_columns=["token_count", "cluster_id", "text"]
    )
    
    collator = DataCollatorWithPadding(
        tokenizer,
        padding='max_length',
        max_length=64,
        pad_to_multiple_of=8,
        return_tensors='pt'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collator,
    )
    
    return dataloader

def main():
    print("Hello from dllm-pretraining!")


if __name__ == "__main__":
    main()
