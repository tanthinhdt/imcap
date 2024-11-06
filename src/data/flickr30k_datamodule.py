import os
import polars as pl
from pathlib import Path
from lightning import LightningDataModule
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor
from PIL import Image


class Flickr30kDataModule(LightningDataModule):
    """
    LightningDataModule for the Flickr30k dataset.
    """

    def __init__(
        self,
        processor: AutoProcessor,
        data_dir: str = "data/flickr30k",
        use_all_comments: bool = False,
        comment_number: int = 0,
        padding: str = "max_length",
        max_length: int = 128,
        truncation: bool = True,
        train_val_test_split: Tuple[float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """
        Initialize a Flickr30kDataModule instance.

        Parameters
        ----------
        processor : AutoProcessor
            The processor to use for tokenization.
        data_dir : str, optional
            The directory where the dataset is stored, by default "data/flickr30k".
        use_all_comments : bool, optional
            Whether to use all comments or a single comment, by default False.
        comment_number : int, optional
            The comment number to select, by default None.
        padding : str, optional
            The padding strategy, by default "max_length".
        max_length : int, optional
            The maximum length of the sequence, by default 128.
        truncation : bool, optional
            Whether to truncate the sequence, by default True.
        train_val_test_split : Tuple[float], optional
            The split ratio for the train, validation, and test sets, by default (0.8, 0.1, 0.1).
        batch_size : int, optional
            The batch size, by default 64.
        num_workers : int, optional
            The number of workers to use for data loading, by default 0.
        pin_memory : bool, optional
            Whether to use pinned memory for data loading, by default False.
        """
        super().__init__()
        assert 1 < len(train_val_test_split) < 4 and 0 < sum(train_val_test_split) <= 1.0, \
            "The sum of train_val_test_split must be 1.0."

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.processor = processor

        self.dataset: Optional[DatasetDict] = None
        self.num_examples: int = 0

        self.batch_size_per_device = batch_size

    def __len__(self) -> int:
        """
        Return the number of examples in the dataset.

        Returns
        -------
        int
            The number of examples in the dataset.
        """
        return self.num_examples

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data and split it into training, validation, and test sets.

        Parameters
        ----------
        stage : Optional[str], optional
            The stage to setup, by default
        """
        def transform(batch):
            images = [
                Image.open(Path(self.hparams.data_dir) / f"flickr30k_images/{image_name}")
                for image_name in batch["image_name"]
            ]
            texts = [comment.strip() for comment in batch["comment"]]
            batch = self.processor(
                images=images,
                text=texts,
                padding=self.hparams.padding,
                max_length=self.hparams.max_length,
                truncation=self.hparams.truncation,
                return_tensors="pt",
            )
            batch.update({"labels": batch["input_ids"]})
            return batch

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
            if not self.hparams.use_all_comments:
                df = df.filter(pl.col("comment_number") == self.hparams.comment_number)
            self.num_examples = len(df)
            dataset = Dataset.from_polars(df)
            dataset = dataset.shuffle(seed=int(os.environ.get("PL_GLOBAL_SEED", 42)))
            if len(self.hparams.train_val_test_split) == 2:
                self.dataset = dataset.train_test_split(
                    train_size=self.hparams.train_val_test_split[0],
                    test_size=self.hparams.train_val_test_split[1],
                    shuffle=False,
                )
            else:
                self.dataset = dataset.train_test_split(
                    train_size=sum(self.hparams.train_val_test_split[:2]),
                    test_size=self.hparams.train_val_test_split[2],
                    shuffle=False,
                )
                train_val_splits = self.dataset["train"].train_test_split(
                    train_size=self.hparams.train_val_test_split[0],
                    shuffle=False,
                )
                self.dataset = DatasetDict({
                    "train": train_val_splits["train"],
                    "val": train_val_splits["test"],
                    "test": self.dataset["test"],
                })
            self.dataset.set_transform(transform)

    def train_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the train dataloader.

        Returns
        -------
        DataLoader[Any]
            The train dataloader.
        """
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the validation dataloader.

        Returns
        -------
        DataLoader[Any]
            The validation dataloader.
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
        """
        Create and return the test dataloader.

        Returns
        -------
        DataLoader[Any]
            The test dataloader.
        """
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Lightning hook for cleaning up after trainer.fit(), trainer.validate(),
        trainer.test(), and `trainer.predict().

        Parameters
        ----------
        stage : Optional[str], optional
            The stage to teardown, by default
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """
        Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns
        -------
        Dict[Any, Any]
            The datamodule state to be saved.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        state_dict().

        Parameters
        ----------
        state_dict : Dict[str, Any]
            The datamodule state to be loaded.
        """
        pass


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        cache_dir="models/huggingface",
    )
    dm = Flickr30kDataModule(processor=processor)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    print(batch.keys())
