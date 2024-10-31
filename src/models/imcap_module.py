import torch
from typing import Any, Dict, Tuple
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.text import WordErrorRate, BLEUScore
from transformers import AutoModelForVision2Seq, AutoProcessor


class IMCAPLitModule(LightningModule):
    """
    Example of a LightningModule for Image Captioning.
    """

    def __init__(
        self,
        net: AutoModelForVision2Seq,
        processor: AutoProcessor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        hf_repo_id: str = None,
    ) -> None:
        """
        Initialize a IMCAPLitModule.

        Parameters
        ----------
        net : AutoModelForVision2Seq
            The model to use for training.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training.
        scheduler : torch.optim.lr_scheduler
            The learning rate scheduler to use for training.
        compile : bool
            Whether to compile the model before training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.processor = processor

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_wer = WordErrorRate()
        self.train_bleu = BLEUScore()
        self.val_wer = WordErrorRate()
        self.val_bleu = BLEUScore()
        self.test_wer = WordErrorRate()
        self.test_bleu = BLEUScore()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far metrics
        self.val_wer_best = MaxMetric()
        self.val_bleu_best = MaxMetric()
        self.test_wer_best = MaxMetric()
        self.test_bleu_best = MaxMetric()

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_wer.reset()
        self.val_bleu.reset()
        self.val_wer_best.reset()
        self.val_bleu_best.reset()

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch of data containing the input tensor of images and target labels.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        outputs = self.net(**batch)
        preds = self.processor.batch_decode(
            torch.argmax(outputs.logits, dim=-1),
            skip_special_tokens=True,
        )
        targets = self.processor.batch_decode(
            batch["labels"],
            skip_special_tokens=True,
        )
        return outputs.loss, preds, targets

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step on a batch of data from the training set.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch of data containing the input tensor of images and target labels.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The loss value for the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_wer(preds, targets)
        self.log("train/wer", self.train_wer, on_step=False, on_epoch=True, prog_bar=True)
        self.train_bleu(preds, [target for target in targets])
        self.log("train/bleu", self.train_bleu, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Lightning hook that is called when a training epoch ends.
        """
        if self.hparams.hf_repo_id is not None:
            self.net.push_to_hub(self.hparams.hf_repo_id)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Perform a single validation step on a batch of data from the validation set.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch of data containing the input tensor of images and target labels.
        batch_idx : int
            The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_wer(preds, targets)
        self.log("val/wer", self.val_wer, on_step=False, on_epoch=True, prog_bar=True)
        self.val_bleu(preds, [target for target in targets])
        self.log("val/bleu", self.val_bleu, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Lightning hook that is called when a validation epoch ends.
        """
        wer = self.val_wer.compute()
        self.val_wer_best(wer)
        self.log("val/wer_best", self.val_wer_best.compute(), sync_dist=True, prog_bar=True)

        bleu = self.val_bleu.compute()
        self.val_bleu_best(bleu)
        self.log("val/bleu_best", self.test_bleu_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Perform a single test step on a batch of data from the test set.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A batch of data (a tuple) containing the input tensor of images and target labels.
        batch_idx : int
            The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_wer(preds, targets)
        self.log("test/wer", self.test_wer, on_step=False, on_epoch=True, prog_bar=True)
        self.test_bleu(preds, [target for target in targets])
        self.log("test/bleu", self.test_bleu, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """
        Lightning hook that is called when a test epoch ends.
        """
        wer = self.test_wer.compute()
        self.test_wer_best(wer)
        self.log("test/wer_best", self.test_wer_best.compute(), sync_dist=True, prog_bar=True)

        bleu = self.test_bleu.compute()
        self.test_bleu_best(bleu)
        self.log("test/bleu_best", self.test_bleu_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """
        Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Parameters
        ----------
        stage : str
            The stage for which the setup is being performed.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the optimizer and learning-rate scheduler.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = IMCAPLitModule(None, None, None, None)
