from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
import json
from typing import Any, Callable, Self
import uuid

import numpy as np


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

    array: np.array
    shape: tuple[int, int]
    depth: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_array(cls, array: np.array, metadata: dict | None = None) -> Self:
        metadata = metadata or {}
        return cls(
            array=array,
            shape=array.shape,
            depth=str(array.dtype),
            metadata=metadata,
        )

    @classmethod
    def decode(
            cls,
            encoded_image: bytes,
            separator: bytes | None = None,
            deserializer: Callable = _deserializer,
    ) -> Self:
        separator = separator or cls._separator
        elems = encoded_image.split(separator, maxsplit=4)
        depth = elems[2].decode("utf8")
        array = np.frombuffer(elems[0], dtype=depth)
        shape_info = elems[1].decode("utf8").split(",")
        shape = (int(shape_info[0]), int(shape_info[2]))
        array.reshape(shape)
        return cls(
            array=array,
            shape=shape,
            depth=depth,
            metadata=deserializer(elems[3])
        )

    def encode(
            self,
            separator: bytes | None = None,
            serializer: Callable = _serializer,
    ) -> bytes:
        """
        :param separator: A bytes used as field separator in the bytes payload
        :param serializer: A function that takes an object and returns bytes
        :return: The image and its metadata encoded as a bytes payload
        """
        separator = separator or self._separator
        return (
            # b"image" + separator +
            self.array.tobytes() + separator +
            f"{self.shape[0]},{self.shape[1]}".encode("utf8") + separator +
            self.depth.encode("utf8") + separator +
            serializer(self.metadata)
        )
