from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import json
from typing import Any, Self
import uuid

import numpy as np
from PIL import Image as PIL_image


def _default(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.astimezone(tz=timezone.utc).isoformat(timespec="seconds")
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, time):
        return str(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)


class _Serializer:
    @staticmethod
    def dumps(obj: Any) -> bytes:
        return json.dumps(obj, default=_default).encode("utf8")

    @staticmethod
    def loads(obj: Any) -> Any:
        return json.loads(obj.decode("utf8"))


class SerializableImage:
    _separator = b"\x1f\x1f"
    _serializer = _Serializer

    def __init__(
            self,
            byte_array: bytes,
            shape: tuple[int, ...],
            depth: str,
            metadata: dict | None = None,
    ):
        self.byte_array: bytes = byte_array
        self._array: np.ndarray | None = None
        self.shape: tuple[int, ...] = shape
        self.depth: str = depth
        self.metadata: dict = metadata or {}

    def __repr__(self) -> str:
        return f"<SerializableImage(shape={self.shape}, depth={self.depth})>"

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            array = np.frombuffer(self.byte_array, dtype=self.depth)
            array = array.reshape(self.shape)
            self._array = array
        return self._array

    @classmethod
    def from_array(cls, array: np.ndarray, metadata: dict | None = None) -> Self:
        """Create an Image from a numpy array

        :param array: An image as numpy array
        :param metadata: Information about the image
        :return: A SerializableImage
        """
        metadata = metadata or {}
        byte_array = array.tobytes()
        obj = cls(
            byte_array=byte_array,
            shape=array.shape,
            depth=str(array.dtype),
            metadata=metadata,
        )
        obj._array = array
        return obj

    @classmethod
    def from_image(cls, image: PIL_image.Image, metadata: dict | None = None) -> Self:
        """Create an Image from a numpy array

        :param image: An image as a PIL Image
        :param metadata: Information about the image
        :return: A SerializableImage
        """
        array = np.array(image)
        return cls.from_array(array, metadata)

    @classmethod
    def decode(cls, encoded_image: bytes) -> Self:
        """Decode the bytes payload containing an Image and return it

        :param encoded_image: An Image encoded into bytes
        :return: A SerializableImage with the info from the payload
        """
        elems = encoded_image.split(cls._separator, maxsplit=4)
        byte_array = elems[0]
        shape_info = elems[1].decode("utf8").split(",")
        shape = tuple([int(dim) for dim in shape_info])
        depth = elems[2].decode("utf8")
        return cls(
            byte_array=byte_array,
            shape=shape,
            depth=depth,
            metadata=cls._serializer.loads(elems[3])
        )

    def encode(self) -> bytes:
        """Encode the Image as a bytes payload to send it via a dispatcher

        :return: A SerializableImage and its metadata encoded as a bytes payload
        """
        return (
            # b"image" + separator +
            self.byte_array + self._separator +
            ",".join([str(dim) for dim in self.shape]).encode("utf8") + self._separator +
            self.depth.encode("utf8") + self._separator +
            self._serializer.dumps(self.metadata)
        )


class SerializableImagePayload:
    _separator = b"\x1e\x1e"

    uid: str
    data: list[SerializableImage]

    def __init__(
            self,
            uid: str,
            data: list[SerializableImage],
    ):
        self.uid: str = uid
        self.data: list[SerializableImage] = data

    def __repr__(self) -> str:
        return f"<SerializableImagePayload({self.uid}, elements={len(self.data)})>"

    @classmethod
    def decode(cls, encoded_payload: bytes) -> Self:
        """Decode the bytes payload containing an Image payload and return it

        :param encoded_payload: An Image encoded into bytes
        :return: A SerializableImagePayload with the data from the payload
        """
        elems = encoded_payload.split(cls._separator)
        return cls(
            uid=elems[0].decode("utf8"),
            data=[SerializableImage.decode(data) for data in elems[1:]]
        )

    def encode(self) -> bytes:
        """Encode the Images payload as a bytes payload to send it via a dispatcher

        :return: A SerializableImage and its metadata encoded as a bytes payload
        """
        return (
            self.uid.encode("utf8") + self._separator +
            self._separator.join([image.encode() for image in self.data])
        )
