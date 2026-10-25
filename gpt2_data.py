import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import GPT2Tokenizer
from datasets import load_dataset


class WikiText2Data:
    def __init__(self, batch_size, num_workers, seq_len=128, calib_size=128, calib_seed=42):
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.seq_len     = seq_len
        self.calib_seed  = calib_seed

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

        self.train_blocks = self._make_blocks(dataset["train"]["text"],       "train")
        self.val_blocks   = self._make_blocks(dataset["validation"]["text"],  "validation")
        self.test_blocks  = self._make_blocks(dataset["test"]["text"],        "test")

        rng            = torch.Generator().manual_seed(calib_seed)
        self.calib_idx = torch.randperm(len(self.train_blocks), generator=rng)[:calib_size]

    def _make_blocks(self, texts, split_name):
        sep = self.tokenizer.eos_token_id
        ids = []
        for t in texts:
            t = t.strip()
            if t:
                ids.extend(self.tokenizer.encode(t))
                ids.append(sep)

        n_blocks  = len(ids) // self.seq_len
        discarded = len(ids) - n_blocks * self.seq_len
        print(f"{split_name}: {n_blocks} blocks, {discarded} tokens discarded")

        return torch.tensor(ids[:n_blocks * self.seq_len], dtype=torch.long).view(n_blocks, self.seq_len)

    def calib_dataloader(self):
        subset = Subset(TensorDataset(self.train_blocks), self.calib_idx.tolist())
        return DataLoader(subset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.train_blocks),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.val_blocks),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            TensorDataset(self.test_blocks),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
