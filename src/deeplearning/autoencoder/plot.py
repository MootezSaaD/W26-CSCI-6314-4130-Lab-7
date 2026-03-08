import matplotlib.pyplot as plt
from typing import Optional

from trainer import Trainer

def plot_losses(trainer: Trainer) -> None:
    """
    Plots training and validation losses with smoothing for better visualization
    
    Args:
        trainer: Trainer instance containing training history
        window_size: Size of moving average window for smoothing (1 = no smoothing)
    """
    plt.figure(figsize=(10, 6))
    
    # Smoothing function
    def _smooth(scalars: list[float], weight: float) -> list[float]:
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    # Plot training loss
    train_loss = trainer.history['train_loss']
    if train_loss:
        epochs = range(1, len(train_loss) + 1)
        smoothed_train = _smooth(train_loss, 0.6)  # 0.6 smoothing weight
        plt.plot(epochs, smoothed_train, 'b-', alpha=0.3, label='Training Loss (smoothed)')
        plt.plot(epochs, train_loss, 'b.', alpha=0.2)
    
    # Plot validation loss
    val_loss = trainer.history['val_loss']
    if val_loss:
        smoothed_val = _smooth(val_loss, 0.6)
        plt.plot(epochs, smoothed_val, 'r-', alpha=0.3, label='Validation Loss (smoothed)')
        plt.plot(epochs, val_loss, 'r.', alpha=0.2)
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Set x-axis ticks to integer values
    if len(train_loss) > 0:
        plt.xticks(range(1, len(train_loss) + 1))
    
    plt.show()

# Usage example:
# plot_losses(trainer)