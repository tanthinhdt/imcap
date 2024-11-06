import pytest
from pathlib import Path
from transformers import AutoProcessor
from src.data import Flickr30kDataModule


parameters = ["batch_size", "train_val_test_split"]
values = [
    (32, (0.8, 0.1, 0.1)),
    (32, (0.8, 0.2)),
    (128, (0.8, 0.1, 0.1)),
    (128, (0.8, 0.2)),
]


@pytest.mark.parametrize(parameters, values)
def test_flickr30k_datamodule(batch_size: int, train_val_test_split: tuple) -> None:
    """
    Tests Flickr30kDataModule to verify that:
    - The dataset is loaded correctly.
    - The dataset is split into training, validation, and testing sets correctly.
    - The dataloaders are created correctly.
    - The batch size of the dataloaders is correct.
    - The shapes of the data in the batch are correct.

    Parameters
    ----------
    batch_size : int
        The batch size to use for the dataloaders.
    train_val_test_split : tuple
        A tuple containing the train, validation, and test split ratios.
    """
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        cache_dir="models/huggingface",
    )
    data_dir = "data/flickr30k"
    use_all_comments = False
    comment_number = 0
    padding = "max_length"
    max_length = 128
    truncation = True
    crop_size = 384

    assert Path(data_dir, "results.csv").exists(), \
        "Metadata file does not exist. Make sure to include the metadata file."
    assert Path(data_dir, "flickr30k_images").exists(), \
        "Image directory does not exist. Make sure to include the image directory."

    dm = Flickr30kDataModule(
        processor=processor,
        data_dir=data_dir,
        use_all_comments=use_all_comments,
        comment_number=comment_number,
        padding=padding,
        max_length=max_length,
        truncation=truncation,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
    )
    assert not dm.dataset, \
        "Dataset should not be loaded before calling `setup`."

    dm.setup()
    if len(train_val_test_split) == 3:
        assert "train" in dm.dataset and "val" in dm.dataset and "test" in dm.dataset, \
            "Dataset should be split into training, validation and testing sets."
        num_datapoints = len(dm.dataset["train"]) + len(dm.dataset["val"]) + len(dm.dataset["test"])
    else:
        assert "train" in dm.dataset and "test" in dm.dataset, \
            "Dataset should be split into training and testing sets."
        num_datapoints = len(dm.dataset["train"]) + len(dm.dataset["test"])
    assert num_datapoints == int(len(dm) * sum(train_val_test_split)), \
        (
            f"Dataset size mismatch. Expected {int(len(dm) * sum(train_val_test_split))}, "
            f"but got {num_datapoints}."
        )

    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader(), \
        "Dataloaders should be created after calling `setup`."
    batch = next(iter(dm.train_dataloader()))

    assert batch["pixel_values"].shape == (batch_size, 3, crop_size, crop_size), \
        (
            f"Images shape mismatch. Expected {(batch_size, 3, crop_size, crop_size)}, "
            f"but got {batch['pixel_values'].shape}."
        )
    assert batch["input_ids"].shape == (batch_size, max_length), \
        (
            f"Input IDs shape mismatch. Expected {(batch_size, max_length)}, "
            f"but got {batch['input_ids'].shape}."
        )
    assert batch["attention_mask"].shape == (batch_size, max_length), \
        (
            f"Attention Mask shape mismatch. Expected {(batch_size, max_length)}, "
            f"but got {batch['attention_mask'].shape}."
        )
    assert batch["labels"].shape == (batch_size, max_length), \
        (
            f"Labels shape mismatch. Expected {(batch_size, max_length)}, "
            f"but got {batch['labels'].shape}."
        )
