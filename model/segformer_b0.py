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
            MixTransformerEncoderLayer(in_channels=b0_config['mix_transformer_encoder_layer_1']['in_channels'], out_channels=b0_config['mix_transformer_encoder_layer_1']['out_channels'], patch_size=b0_config['mix_transformer_encoder_layer_1']['patch_size'], stride=b0_config['mix_transformer_encoder_layer_1']['stride'], padding=b0_config['mix_transformer_encoder_layer_1']['padding'], n_layers=b0_config['mix_transformer_encoder_layer_1']['n_layers'], reduction_ratio=b0_config['mix_transformer_encoder_layer_1']['reduction_ratio'], num_heads=b0_config['mix_transformer_encoder_layer_1']['num_heads'], expansion_factor=b0_config['mix_transformer_encoder_layer_1']['expansion_factor']),
            MixTransformerEncoderLayer(in_channels=b0_config['mix_transformer_encoder_layer_2']['in_channels'], out_channels=b0_config['mix_transformer_encoder_layer_2']['out_channels'], patch_size=b0_config['mix_transformer_encoder_layer_2']['patch_size'], stride=b0_config['mix_transformer_encoder_layer_2']['stride'], padding=b0_config['mix_transformer_encoder_layer_2']['padding'], n_layers=b0_config['mix_transformer_encoder_layer_2']['n_layers'], reduction_ratio=b0_config['mix_transformer_encoder_layer_2']['reduction_ratio'], num_heads=b0_config['mix_transformer_encoder_layer_2']['num_heads'], expansion_factor=b0_config['mix_transformer_encoder_layer_2']['expansion_factor']),
            MixTransformerEncoderLayer(in_channels=b0_config['mix_transformer_encoder_layer_3']['in_channels'], out_channels=b0_config['mix_transformer_encoder_layer_3']['out_channels'], patch_size=b0_config['mix_transformer_encoder_layer_3']['patch_size'], stride=b0_config['mix_transformer_encoder_layer_3']['stride'], padding=b0_config['mix_transformer_encoder_layer_3']['padding'], n_layers=b0_config['mix_transformer_encoder_layer_3']['n_layers'], reduction_ratio=b0_config['mix_transformer_encoder_layer_3']['reduction_ratio'], num_heads=b0_config['mix_transformer_encoder_layer_3']['num_heads'], expansion_factor=b0_config['mix_transformer_encoder_layer_3']['expansion_factor']),
            MixTransformerEncoderLayer(in_channels=b0_config['mix_transformer_encoder_layer_4']['in_channels'], out_channels=b0_config['mix_transformer_encoder_layer_4']['out_channels'], patch_size=b0_config['mix_transformer_encoder_layer_4']['patch_size'], stride=b0_config['mix_transformer_encoder_layer_4']['stride'], padding=b0_config['mix_transformer_encoder_layer_4']['padding'], n_layers=b0_config['mix_transformer_encoder_layer_4']['n_layers'], reduction_ratio=b0_config['mix_transformer_encoder_layer_4']['reduction_ratio'], num_heads=b0_config['mix_transformer_encoder_layer_4']['num_heads'], expansion_factor=b0_config['mix_transformer_encoder_layer_4']['expansion_factor']),
        ])
        # Define the decoder
        self.decoder = MLPDecoder(in_channels=b0_config['mlp_decoder']['in_channels'], embed_channels=b0_config['mlp_decoder']['embed_channels'], out_dims=b0_config['mlp_decoder']['out_dims'], num_classes=b0_config['mlp_decoder']['num_classes'])
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=b0_config['optimizer']['learning_rate'])
        return optimizer
    
    def miou(self, prediction, targets):
        # Calculate mean intersection over union (mIoU) for segmentation accuracy
        thr = b0_config['miou']['threshold']
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
