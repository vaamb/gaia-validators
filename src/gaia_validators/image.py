from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
import json
from pathlib import Path
from typing import Any, Callable, Self
import uuid

import numpy as np
from PIL import Image as PILImage


def _default(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.astimezone(tz=timezone.utc).isoformat(timespec="seconds")
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, time):
        return str(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)


def _serializer(obj: Any) -> bytes:
    return json.dumps(obj, default=_default).encode("utf8")


def _deserializer(payload: bytes) -> Any:
    return json.loads(payload.decode("utf8"))


@dataclass
class Image:
    _separator = b"\-"
    _serializer = _serializer
    _deserializer = _deserializer

    array: np.array
    shape: tuple[int, int]
    depth: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_array(cls, array: np.array, metadata: dict | None = None) -> Self:
        """Create an Image from a numpy array

        :param array: An image as numpy array
        :param metadata: Information about the image
        :return: The Image
        """
        metadata = metadata or {}
        return cls(
            array=array,
            shape=array.shape,
            depth=str(array.dtype),
            metadata=metadata,
        )

    @classmethod
    def decode(cls, encoded_image: bytes) -> Self:
        """Decode the bytes payload containing an Image and return it

        :param encoded_image: An Image encoded into bytes
        :return: The Image with the info from the payload
        """
        elems = encoded_image.split(cls._separator, maxsplit=4)
        depth = elems[2].decode("utf8")
        array = np.frombuffer(elems[0], dtype=depth)
        shape_info = elems[1].decode("utf8").split(",")
        shape = (int(shape_info[0]), int(shape_info[2]))
        array.reshape(shape)
        return cls(
            array=array,
            shape=shape,
            depth=depth,
            metadata=cls._deserializer(elems[3])
        )

    def encode(self) -> bytes:
        """Encode the Image as a bytes payload to send it via a dispatcher

        :return: The Image and its metadata encoded as a bytes payload
        """
        return (
            # b"image" + separator +
            self.array.tobytes() + self._separator +
            f"{self.shape[0]},{self.shape[1]}".encode("utf8") + self._separator +
            self.depth.encode("utf8") + self._separator +
            self._serializer(self.metadata)
        )

    @classmethod
    def open(cls, path: Path, metadata: dict | None = None) -> Self:
        """Open and return a file image

        :param path: The Path from where the Image should be opened
        :param metadata: The image metadata
        :return: The Image
        """
        metadata = metadata or {}
        pil_image = PILImage.open(path)
        array = np.asarray(pil_image)
        return cls(
            array=array,
            shape=array.shape,
            depth=str(array.dtype),
            metadata=metadata
        )

    def save(self, name: Path, extension: str = ".jpeg") -> Path:
        """Save the Image as a file

        :param name: The Path where the Image should be saved
        :param extension: The image file extension to use
        :return: The Path where the Image has been saved
        """
        pil_image = PILImage.fromarray(self.array)
        if not str(name).endswith(extension):
            name = name/extension
        pil_image.save(name)
        return name
