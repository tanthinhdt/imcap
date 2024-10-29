import torch
import polars as pl
from lightning import LightningDataModule
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from torchvision.transforms import v2 as transforms
from pathlib import Path
from PIL import Image


class Flickr30kDataModule(LightningDataModule):
    """`LightningDataModule` for the Flickr30k dataset.
    """

    def __init__(
        self,
        data_dir: str = "data/flickr30k",
        tokenizer: str = "bert-base-uncased",
        train_val_test_split: Tuple[float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize a Flickr30kDataModule instance.

        Parameters
        ----------
        data_dir : str, optional
            The directory where the dataset is stored, by default "data/flickr30k".
        train_val_test_split : Tuple[float], optional
            The split ratio for the train, validation, and test sets, by default (0.8, 0.1, 0.1).
        batch_size : int, optional
            The batch size, by default 64.
        num_workers : int, optional
            The number of workers to use for data loading, by default 0.
        pin_memory : bool, optional
            Whether to use pinned memory for data loading, by default False.
        seed : int, optional
            The seed to use for reproducibility, by default 42.
        """
        super().__init__()
        assert sum(train_val_test_split) == 1.0, \
            "The sum of train_val_test_split must be 1.0."

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dataset: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.
        """
        def transform(batch):
            images = [
                Image.open(Path(self.hparams.data_dir) / "flickr30k_images" / image_name)
                for image_name in batch["image_name"]
            ]
            images = [self.transforms(image) for image in images]

            comments = self.tokenizer(
                text=[s.strip() for s in batch["comment"]],
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt",
            )

            return {
                "images": images,
                "input_ids": comments["input_ids"],
                "attention_mask": comments["attention_mask"],
                "token_type_ids": comments["token_type_ids"],
                "comment_number": batch["comment_number"],
            }

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible "
                    "by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.dataset:
            df = pl.read_csv(
                f"{self.hparams.data_dir}/results.csv",
                separator="|",
                schema_overrides={
                    "image_name": pl.String,
                    "comment_number": pl.Int32,
                    "comment": pl.String,
                },
            )
            self.dataset = Dataset.from_polars(df)
            self.dataset = self.dataset.train_test_split(
                test_size=self.hparams.train_val_test_split[2],
                shuffle=True,
                seed=self.hparams.seed,
            )
            self.dataset.set_transform(transform)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        data_val = self.dataset["val"] if "val" in self.dataset else self.dataset["test"]
        return DataLoader(
            dataset=data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = Flickr30kDataModule()
