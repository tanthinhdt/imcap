import pytest
from pathlib import Path
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
    data_dir = "data/"
    processor = "Salesforce/blip-image-captioning-base"
    image_size = 384
    max_length = 128

    data_dir = Path(data_dir)
    assert data_dir.exists(), \
        f"{data_dir} does not exist. Make sure to download the dataset."

    dataset_dir = data_dir / "flickr30k"
    assert dataset_dir.exists(), \
        f"{dataset_dir} does not exist. Make sure to download the dataset."
    assert Path(dataset_dir, "results.csv").exists(), \
        "Metadata file does not exist. Make sure to include the metadata file."
    assert Path(dataset_dir, "flickr30k_images").exists(), \
        "Image directory does not exist. Make sure to include the image directory."

    dm = Flickr30kDataModule(
        data_dir=dataset_dir,
        processor=processor,
        max_length=max_length,
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
    assert num_datapoints == int(158_915 * sum(train_val_test_split)), \
        (
            f"Dataset size mismatch. Expected {int(158_915 * sum(train_val_test_split))}, "
            f"but got {num_datapoints}."
        )

    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader(), \
        "Dataloaders should be created after calling `setup`."
    batch = next(iter(dm.train_dataloader()))

    assert batch["pixel_values"].shape == (batch_size, 3, image_size, image_size), \
        (
            f"Images shape mismatch. Expected {(batch_size, 3, image_size, image_size)}, "
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
