from .decoder import MLPDecoder
from .encoder import MixTransformerEncoderLayer

import pytorch_lightning as pl
import torch
import torch.nn as nn

class SegFormer_B0(pl.LightningModule):
    """
    SegFormer_B0 for image segmentation tasks.

    This class implements a variant of the SegFormer architecture using PyTorch Lightning. It includes a series of transformer encoder layers for feature extraction and an MLP-based decoder for segmentation. The class also contains methods for training, validation, and custom logging.

    Attributes:
        encoder_blocks (nn.ModuleList): List of transformer encoder layers for feature extraction.
        decoder (MLPDecoder): MLP-based decoder for segmentation.
        loss (nn.CrossEntropyLoss): Loss function for training.
        _lossLogs (list): List to store loss logs.
        _miouLogs (list): List to store mean intersection over union (mIoU) logs.
        losslogsPerBatch (float): Accumulator for loss per batch.
        mioulogsPerBatch (float): Accumulator for mIoU per batch.
        trainCounter (int): Counter for training batches.
        valCounter (int): Counter for validation batches.
    """

    def __init__(self):
        super().__init__()
        # Define encoder blocks with varying parameters
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(3, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8),
            MixTransformerEncoderLayer(64, 160, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=2, num_heads=5, expansion_factor=4),
            MixTransformerEncoderLayer(160, 256, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=1, num_heads=8, expansion_factor=4)
        ])
        # Define the decoder
        self.decoder = MLPDecoder([32, 64, 160, 256], 256, (64, 64), 4)
        # Loss function
        self.loss = nn.CrossEntropyLoss()

        # Custom logging setup
        self._lossLogs = []
        self._miouLogs = []
        self.losslogsPerBatch = 0
        self.mioulogsPerBatch = 0
        self.trainCounter = 0
        self.valCounter = 0
    
    def forward(self, images):
        # Process images through encoder blocks and collect embeddings
        embeds = [images]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        # Decode the embeddings
        return self.decoder(embeds[1:])
        
    def configure_optimizers(self):
        # Configure optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer
    
    def miou(self, prediction, targets):
        # Calculate mean intersection over union (mIoU) for segmentation accuracy
        thr = 1e-5
        C = prediction.size()[1]
        pred = prediction.argmax(dim=1)
        validClass = C
        iou = 0

        for i in range(C):
            maskTar = targets == i
            maskPred = pred == i
            
            if maskTar.sum() == 0:
                validClass -= 1
                continue

            intersect = torch.logical_and(maskPred, maskTar).sum().item()
            union = torch.logical_or(maskPred, maskTar).sum().item()
            iou += intersect / (union + thr)

        iou = iou / validClass
        return iou

    def training_step(self, batch, batch_idx):
        # Training step
        images, targets = batch
        predictions = self.forward(images)
        loss = self.loss(predictions, targets)
        self.log('train_loss', loss, prog_bar=True)

        # Update loss logs
        self.losslogsPerBatch += loss.detach().cpu()
        self.trainCounter += 1

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Validation step
        images, targets = batch
        predictions = self.forward(images)
        miou = self.miou(predictions, targets)
        self.log('miou', miou, prog_bar=True)

        # Update mIoU logs
        self.mioulogsPerBatch += miou
        self.valCounter += 1

    def on_train_epoch_end(self):
        # Log and reset training metrics at the end of each epoch
        self._lossLogs.append(self.losslogsPerBatch / self.trainCounter)
        self.trainCounter = 0
        self.losslogsPerBatch = 0
        torch.save(self._lossLogs, 'savedVars/LossPerEp.pt')

    def on_validation_epoch_end(self):
        # Log and reset validation metrics at the end of each epoch
        self._miouLogs.append(self.mioulogsPerBatch / self.valCounter)
        self.valCounter = 0
        self.mioulogsPerBatch = 0
        torch.save(self._miouLogs, 'savedVars/miouPerEp.pt')
