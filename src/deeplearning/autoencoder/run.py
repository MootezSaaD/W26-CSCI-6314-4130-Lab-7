from trainer import Trainer
from model import TransformerAutoEncoder
from dataset import MemoryDataset
from plot import plot_losses
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer AutoEncoder")
    parser.add_argument("--model_tag", type=str, default="microsoft/codebert-base", help="Model tag for AutoTokenizer and AutoModel")
    parser.add_argument("--bottleneck_dim", type=int, default=128, help="Bottleneck dimension")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training dataset JSONL")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation dataset JSONL")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset JSONL")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth", help="Path to save the best model checkpoint")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_tag)
    encoder = AutoModel.from_pretrained(args.model_tag)
    model = TransformerAutoEncoder(
        hidden_dim=encoder.config.hidden_size,
        bottleneck_dim=args.bottleneck_dim,
        bert_encoder=encoder
    )

    ## Freeze the CodeBERT encoder
    for param in model.bert_encoder.parameters():
        param.requires_grad = False

    train_dataset = MemoryDataset(args.train_data, 256, tokenizer)
    validation_dataset = MemoryDataset(args.val_data, 256, tokenizer)
    test_dataset = MemoryDataset(args.test_data, 256, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    trainer = Trainer(model, optimizer, loss_fn, epochs=args.epochs)

    trainer.fit(train_dataloader, val_dataloader, checkpoint_path=args.checkpoint_path)
    trainer.test(test_dataloader)
    plot_losses(trainer.history)