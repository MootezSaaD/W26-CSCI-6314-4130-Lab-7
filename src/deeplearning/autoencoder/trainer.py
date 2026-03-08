import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Callable
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epochs: int = 100,
        verbose: bool = True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.epochs = epochs
        self.verbose = verbose
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
        self.best_metric = np.inf
        self.no_improve = 0

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training", disable=not self.verbose)
        for batch in progress_bar:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            batch_size = inputs['attention_mask'].size(0)
            self.optimizer.zero_grad()
            embedding, outputs = self.model(**inputs)
            loss = self.loss_fn(outputs, embedding)
            loss.backward()

            self.optimizer.step()
            batch_loss = loss.item() * batch_size
            total_loss = total_loss + batch_loss
            self.history['train_loss'].append(loss.item())
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss

    def eval_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", disable=not self.verbose)
            for batch in progress_bar:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                batch_size = inputs['attention_mask'].size(0)
                
                embedding, outputs = self.model(**inputs)
                loss = self.loss_fn(outputs, embedding)
                batch_loss = loss.item() * batch_size

                total_loss += batch_loss
                self.history['val_loss'].append(loss.item())
                progress_bar.set_postfix({'val_loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_path: Optional[str] = None
    ) -> None:
        """
        Full training cycle without early stopping
        Trains for exactly self.epochs number of epochs
        """
        for epoch in range(self.epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss = np.inf
            if val_loader:
                val_loss = self.eval_epoch(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Progress reporting
            if self.verbose:
                epoch_report = [
                    f"Epoch {epoch+1}/{self.epochs}",
                    f"Train Loss: {train_loss:.4f}"
                ]
                if val_loader:
                    epoch_report.append(f"Val Loss: {val_loss:.4f}")
                
                print(" - ".join(epoch_report))

            # Checkpoint saving
            current_metric = val_loss if val_loader else train_loss
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)
                    if self.verbose:
                        print(f"Saved new best model to {checkpoint_path} with Loss: {current_metric:.4f}")

    def test(self, test_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        test_loss = self.eval_epoch(test_loader)
        if self.verbose:
            print(f"\nTest Loss: {test_loss:.4f}")
        return test_loss

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epochs,
            'history': self.history
        }, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
