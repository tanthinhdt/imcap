import pytest
from pathlib import Path
from src.data import Flickr30kDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_flickr30k_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"
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

    dm = Flickr30kDataModule(data_dir=dataset_dir, batch_size=batch_size)
    assert not dm.dataset, \
        "Dataset should not be loaded before calling `setup`."

    dm.setup()
    assert "train" in dm.dataset and "test" in dm.dataset, \
        "Dataset should be split into training and testing sets."
    if "val" in dm.dataset:
        num_datapoints = len(dm.dataset["train"]) + len(dm.dataset["val"]) + len(dm.dataset["test"])
    else:
        num_datapoints = len(dm.dataset["train"]) + len(dm.dataset["test"])
    assert num_datapoints == 158_915, \
        "The dataset should contain 158,915 datapoints."

    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader(), \
        "Dataloaders should be created after calling `setup`."
    batch = next(iter(dm.train_dataloader()))

    assert batch["images"].shape == (batch_size, 3, 224, 224), \
        (
            f"Images shape mismatch. Expected {(batch_size, 3, 224, 224)}, "
            f"but got {batch['images'].shape}."
        )
    assert batch["input_ids"].shape == (batch_size, 128), \
        (
            f"Input IDs shape mismatch. Expected {(batch_size, 128)}, "
            f"but got {batch['input_ids'].shape}."
        )
    assert batch["attention_mask"].shape == (batch_size, 128), \
        (
            f"Attention Mask shape mismatch. Expected {(batch_size, 128)}, "
            f"but got {batch['input_ids'].shape}."
        )
    assert batch["token_type_ids"].shape == (batch_size, 128), \
        (
            f"Token Type IDs shape mismatch. Expected {(batch_size, 128)}, "
            f"but got {batch['token_type_ids'].shape}."
        )
    assert batch["comment_number"].shape == (batch_size,), \
        (
            f"Comment Number shape mismatch. Expected {(batch_size,)}, "
            f"but got {batch['comment_number'].shape}."
        )
