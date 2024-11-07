from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np
import orjson


class _Serializer:
    @staticmethod
    def dumps(obj: Any) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)

    @staticmethod
    def loads(obj: Any) -> Any:
        return orjson.loads(obj)


class SerializableImage:
    _separator = b"\x1f\x1f"
    _serializer = _Serializer

    def __init__(
            self,
            array: np.ndarray,
            metadata: dict | None = None,
    ):
        self.array: np.ndarray = array
        self._mode: str = "BRG"
        self.metadata: dict = metadata or {}

    def __repr__(self) -> str:
        return f"<SerializableImage(shape={self.shape}, depth={self.depth})>"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def depth(self) -> str:
        return str(self.array.dtype)

    @property
    def size(self) -> int:
        return self.array.size

    # ---------------------------------------------------------------------------
    #   Methods to load from and dump to multiple sources
    # ---------------------------------------------------------------------------
    @classmethod
    def from_array(cls, array: np.ndarray, metadata: dict | None = None) -> Self:
        """Create an Image from a numpy array

        :param array: An image as numpy array
        :param metadata: Information about the image
        :return: A SerializableImage
        """
        return cls(
            array=array,
            metadata=metadata or {},
        )

    @classmethod
    def read(cls, path: Path | str, metadata: dict | None = None) -> Self:
        path: Path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        array = cv2.imread(str(path))
        return cls(
            array=array,
            metadata=metadata or {},
        )

    def write(self, path: Path | str) -> None:
        path: Path = Path(path)
        cv2.imwrite(str(path), self.array)

    @classmethod
    def deserialize(cls, encoded_image: bytes) -> Self:
        """Decode the bytes payload containing an Image and return it

        :param encoded_image: An Image encoded into bytes
        :return: A SerializableImage with the info from the payload
        """
        elems = encoded_image.split(cls._separator, maxsplit=4)
        byte_array = elems[0]
        shape_info = elems[1].decode("utf8").split(",")
        shape = tuple([int(dim) for dim in shape_info])
        depth = elems[2].decode("utf8")
        array = np.frombuffer(byte_array, dtype=depth)
        array = array.reshape(shape)
        return cls(
            array=array,
            metadata=cls._serializer.loads(elems[3])
        )

    def serialize(self) -> bytes:
        """Encode the Image as a bytes payload to send it via a dispatcher

        :return: A SerializableImage and its metadata encoded as a bytes payload
        """
        return (
            # b"image" + self._separator +
            self.array.tobytes() + self._separator +
            ",".join([str(dim) for dim in self.shape]).encode("utf8") + self._separator +
            self.depth.encode("utf8") + self._separator +
            self._serializer.dumps(self.metadata)
        )

    @classmethod
    def load_array(cls, path: Path | str, metadata: dict | None = None) -> Self:
        path: Path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handler:
            array: np.ndarray = np.load(handler)
        return cls(
            array=array,
            metadata=metadata or {},
        )

    def dump_array(self, path: Path | str) -> None:
        with path.open("wb") as handler:
            np.save(handler, self.array)

    # ---------------------------------------------------------------------------
    #   Utility methods
    # ---------------------------------------------------------------------------
    def resize(self, new_shape: tuple[int, ...], inplace: bool = True) -> Self:
        array = cv2.resize(self.array, new_shape)
        if inplace:
            self.array = array
            return self
        else:
            return self.__class__(array, self.metadata)

    def to_grayscale(self, inplace: bool = True) -> Self:
        array = cv2.cvtColor(self.array, cv2.COLOR_BGR2GRAY)
        if inplace:
            self.array = array
            return self
        else:
            return self.__class__(array, self.metadata)

    def compute_mse(self, other: SerializableImage) -> float:
        if not self.shape == other.shape:
            raise ValueError("The two arrays must have the same shape")
        return np.mean(
            (self.array.astype(np.float64) - self.array.astype(np.float64)) ** 2,
            dtype=np.float64,
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
    def deserialize(cls, encoded_payload: bytes) -> Self:
        """Decode the bytes payload containing an Image payload and return it

        :param encoded_payload: An Image encoded into bytes
        :return: A SerializableImagePayload with the data from the payload
        """
        elems = encoded_payload.split(cls._separator)
        return cls(
            uid=elems[0].decode("utf8"),
            data=[SerializableImage.deserialize(data) for data in elems[1:]]
        )

    def serialize(self) -> bytes:
        """Encode the Images payload as a bytes payload to send it via a dispatcher

        :return: A SerializableImage and its metadata encoded as a bytes payload
        """
        return (
            self.uid.encode("utf8") + self._separator +
            self._separator.join([image.serialize() for image in self.data])
        )
