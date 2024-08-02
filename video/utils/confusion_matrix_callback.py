import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix

class ConfusionMatrixCallback(pl.Callback):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def on_validation_epoch_end(self, trainer, pl_module):
        val_preds = []
        val_targets = []

        for batch in trainer.val_dataloaders[0]:
            inputs, targets = batch
            inputs, targets = inputs.to(pl_module.device), targets.to(pl_module.device)
            outputs = pl_module(inputs)
            preds = torch.argmax(outputs, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())

        cm = confusion_matrix(val_targets, val_preds, labels=list(range(self.num_classes)))
        print(f"Confusion Matrix for Epoch {trainer.current_epoch}:\n{cm}")
        print("Rows are the actual classes, columns are predicted classes.")
