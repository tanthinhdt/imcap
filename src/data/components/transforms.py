import torch
import random
from typing import Dict, Union, List
from transformers import AutoProcessor
from PIL import Image
from pathlib import Path


class SelectText:
    """
    Select a text from a list of texts. If an index is not provided, a random text is selected.
    """

    def __init__(self, index: int = None) -> None:
        """
        Initialize the transform.

        Parameters
        ----------
        index : int, optional
            The index of the text to select, by default None.
        """
        self.index = index

    def __call__(self, texts: List[str]) -> str:
        """
        Select a text from a list of texts. If an index is not provided, a random text is selected.

        Parameters
        ----------
        texts : List[str]
            The list of texts to select from.

        Returns
        -------
        str
            The selected text.
        """
        return texts[self.index] if self.index is not None else random.choice(texts)


class Strip:
    """
    Strip leading and trailing whitespaces from a string.
    """

    def __call__(self, text: str) -> str:
        """
        Strip leading and trailing whitespaces from a string.

        Parameters
        ----------
        text : str
            The text to strip.

        Returns
        -------
        str
            The stripped text.
        """
        return text.strip()


class Tokenize:
    """
    Tokenize a string into a list of tokens.
    """

    def __init__(
        self,
        processor: AutoProcessor,
        padding: str = "max_length",
        max_length: int = 128,
        truncation: bool = True,
    ) -> None:
        """
        Initialize the transform.

        Parameters
        ----------
        processor : AutoProcessor
            The processor to use.
        padding : str, optional
            The padding strategy, by default "max_length".
        max_length : int, optional
            The maximum length of the sequence, by default 128.
        truncation : bool, optional
            Whether to truncate the sequence, by default True.
        """
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation

    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize a string into a list of tokens.

        Parameters
        ----------
        text : str
            The text to tokenize.

        Returns
        -------
        List[str]
            The list of tokens.
        """
        return self.processor(
            text=text,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors="pt",
        )


class LoadImage:
    """
    Load an image from a file.
    """

    def __init__(self, image_dir: Union[str, Path] = "") -> None:
        """
        Initialize the transform.

        Parameters
        ----------
        image_dir : Union[str, Path], optional
            The directory where the images are stored, by default "".
        """
        self.image_dir = Path(image_dir)

    def __call__(self, image_path: Union[str, Path]) -> Image:
        """
        Load an image from a file.

        Parameters
        ----------
        image_path : Union[str, Path]
            The path to the image file.

        Returns
        -------
        Image
            The loaded image.
        """
        return Image.open(self.image_dir / image_path)
