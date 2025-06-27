from collections.abc import Callable
from typing import Any
import abc
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader as pil_loader
from torchvision.datasets import ImageFolder

from .errors import DatasetNotFoundError
from .utils import (
    BoundingBox,
    Path,
    array_to_tensor,
    concat_samples,
    disambiguate_timestamp,
    merge_samples,
    path_is_vsi,
)

class NonGeoDataset(Dataset[dict[str, Any]], abc.ABC):
    """Abstract base class for datasets lacking geospatial information.

    This base class is designed for datasets with pre-defined image chips.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index

        Raises:
            IndexError: if index is out of range of the dataset
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: NonGeoDataset
    size: {len(self)}"""


class NonGeoClassificationDataset(NonGeoDataset, ImageFolder):  # type: ignore[misc]
    """Abstract base class for classification datasets lacking geospatial information.

    This base class is designed for datasets with pre-defined image chips which
    are separated into separate folders per class.
    """

    def __init__(
        self,
        root: Path = 'data',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        loader: Callable[[Path], Any] | None = pil_loader,
        is_valid_file: Callable[[Path], bool] | None = None,
    ) -> None:
        """Initialize a new NonGeoClassificationDataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            loader: a callable function which takes as input a path to an image and
                returns a PIL Image or numpy array
            is_valid_file: A function that takes the path of an Image file and checks if
                the file is a valid file
        """
        # When transform & target_transform are None, ImageFolder.__getitem__(index)
        # returns a PIL.Image and int for image and label, respectively
        super().__init__(
            root=root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        # Must be set after calling super().__init__()
        self.transforms = transforms

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)
        sample = {'image': image, 'label': label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.imgs)

    def _load_image(self, index: int) -> tuple[Tensor, Tensor]:
        """Load a single image and its class label.

        Args:
            index: index to return

        Returns:
            the image and class label
        """
        img, label = ImageFolder.__getitem__(self, index)
        array: np.typing.NDArray[np.int_] = np.array(img)
        tensor = torch.from_numpy(array).float()
        # Convert from HxWxC to CxHxW
        tensor = tensor.permute((2, 0, 1))
        label = torch.tensor(label)
        return tensor, label
