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
            format: str = "raw",
    ):
        self.array: np.ndarray = array
        self._mode: str = "BRG"
        self.format: str = format
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

    @property
    def is_compressed(self) -> int:
        return self.format != "raw"

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
        if self.is_compressed:
            raise ValueError("Only uncompressed (raw) arrays can be written")
        cv2.imwrite(str(path), self.array)

    save = write

    @classmethod
    def deserialize(cls, encoded_image: bytes) -> Self:
        """Decode the bytes payload containing an Image and return it

        :param encoded_image: An Image encoded into bytes
        :return: A SerializableImage with the info from the payload
        """
        elems = encoded_image.split(cls._separator, maxsplit=4)
        shape_info = elems[1].decode("utf8").split(",")
        shape = tuple([int(dim) for dim in shape_info])
        depth = elems[2].decode("utf8")
        byte_array = elems[4]
        array = np.frombuffer(byte_array, dtype=depth)
        array = array.reshape(shape)
        return cls(
            array=array,
            metadata=cls._serializer.loads(elems[3]),
            format=elems[0].decode("utf8"),
        )

    def serialize(self, compression_format: str | None = None) -> bytes:
        """Encode the Image as a bytes payload to send it via a dispatcher

        :return: A SerializableImage and its metadata encoded as a bytes payload
        """
        if compression_format is not None:
            new_image = self.compress(compression_format, inplace=False)
            return new_image.serialize(compression_format=None)

        rv = bytearray()
        rv += self.format.encode("utf8")
        rv += self._separator
        rv += ",".join([str(dim) for dim in self.shape]).encode("utf8")
        rv += self._separator
        rv += self.depth.encode("utf8")
        rv += self._separator
        rv += self._serializer.dumps(self.metadata)
        rv += self._separator
        rv += self.array.tobytes()
        return rv

    encode = serialize
    decode = deserialize


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
        path: Path = Path(path)
        with path.open("wb") as handler:
            np.save(handler, self.array)

    # ---------------------------------------------------------------------------
    #   Utility methods
    # ---------------------------------------------------------------------------
    def compress(self, compression_format: str, inplace: bool = False) -> Self:
        if self.is_compressed:
            raise ValueError("Only uncompressed (raw) arrays can be compressed")
        result, array = cv2.imencode(compression_format, self.array)
        if not result:
            raise RuntimeError("Compression failed")
        if inplace:
            self.array = array
            self.format = compression_format
            return self
        else:
            return self.__class__(array, self.metadata, compression_format)

    def uncompress(self, inplace: bool = False) -> Self:
        if not self.is_compressed:
            raise ValueError("Only compressed (non-raw) arrays can be uncompressed")
        array = cv2.imdecode(self.array, cv2.IMREAD_UNCHANGED)
        if inplace:
            self.array = array
            self.format = "raw"
            return self
        else:
            return self.__class__(array, self.metadata, "raw")

    def resize(
            self,
            new_shape: tuple[int, ...] | None = None,
            ratio: float = 1.0,
            inplace: bool = False,
    ) -> Self:
        if new_shape is None and ratio == 1:
            return self
        array = cv2.resize(self.array, new_shape, fx=ratio, fy=ratio)
        if inplace:
            self.array = array
            return self
        else:
            return self.__class__(array, self.metadata)

    def to_grayscale(self, inplace: bool = False) -> Self:
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
            (self.array.astype(np.float64) - other.array.astype(np.float64)) ** 2,
            dtype=np.float64,
        )  # type: ignore

    def apply_rgb_formula(self, formula: str, inplace: bool = False) -> Self:
        if formula.count("/") > 1:
            raise NotImplementedError(
                "Only formulas with at most one division are supported"
            )
        if self.array.dtype != np.uint8:
            raise NotImplementedError("Only uint8 arrays are currently supported")
        if self.array.ndim != 3:
            raise NotImplementedError("Only BGR arrays are supported")
        formula = formula.lower()
        b, g, r = cv2.split(self.array)
        locals_store = {
            "b": b,
            "g": g,
            "r": r,
        }
        if "/" in formula:
            numerator_str, denominator_str = formula.split("/")
            exec(f"numerator = {numerator_str}", globals(), locals_store)
            exec(f"denominator = {denominator_str}", globals(), locals_store)
            numerator = locals_store["numerator"]
            denominator = locals_store["denominator"]
            denominator[denominator == 0] = 1
            new_array = numerator / denominator
        else:
            exec(f"new_array = {formula}", globals(), locals_store)
            new_array = locals_store["new_array"]
        if inplace:
            self.array = new_array
            return self
        else:
            return self.__class__(new_array, self.metadata)


class SerializableImagePayload:
    _separator = b"\x1e\x1f\x1e"

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

    def serialize(self, compression_format: str | None = None) -> bytes:
        """Encode the Images payload as a bytes payload to send it via a dispatcher

        :return: A SerializableImage and its metadata encoded as a bytes payload
        """
        rv = bytearray()
        rv += self.uid.encode("utf8")
        rv += self._separator
        rv += self._separator.join(
            [
                image.serialize(compression_format)
                for image in self.data
            ]
        )
        return rv

    encode = serialize
    decode = deserialize
