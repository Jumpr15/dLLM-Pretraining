import yaml

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

# pass in yaml config file
def prepare_dataloader(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        tokenizer = config['tokenizer']
        dataset = config['dataset']
        
        tokenize_column = config['tokenize_column']
        remove_columns = config['remove_columns']
        
        batch_size = int(config['batch_size'])
        seq_len = int(config['seq_len'])
        
        
    tokenizer = prepare_tokenizer(tokenizer)
    dataset = prepare_dataset(
        dataset,
        tokenizer,
        seq_len,
        tokenize_column=tokenize_column,
        remove_columns=remove_columns
    )
    
    collator = DataCollatorWithPadding(
        tokenizer,
        padding='max_length',
        max_length=seq_len,
        pad_to_multiple_of=8,
        return_tensors='pt'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
    )
    
    return dataloader

def main():
    print("Hello from dllm-pretraining!")


if __name__ == "__main__":
    main()
