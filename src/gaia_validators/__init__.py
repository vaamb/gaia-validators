from __future__ import annotations

from dataclasses import field
from datetime import datetime, time
from enum import Enum, IntFlag
from typing import Any, Literal, TypedDict

from pydantic import validator
from pydantic.dataclasses import dataclass


def get_enum_names(enum: Enum) -> list:
    return [i.name for i in enum]


def safe_enum_from_name(enum: Enum, name: str | Enum) -> Enum:
    if isinstance(name, enum):
        return name
    return {i.name: i for i in enum}[name]


def safe_enum_from_value(enum: Enum, value: str | Enum) -> Enum:
    if isinstance(value, enum):
        return value
    return {i.value: i for i in enum}[value]


@dataclass()
class Empty:
    pass


@dataclass(frozen=True)
class IDs:
    uid: str
    name: str

    def __iter__(self):
        return iter((self.uid, self.name))


# Config
@dataclass()
class BaseInfoConfig:
    engine_uid: str
    uid: str
    name: str
    status: bool = False


class BaseInfoConfigDict(TypedDict):
    uid: str
    name: str
    status: bool


# Management
class ManagementFlags(IntFlag):
    sensors = 1
    light = 2
    climate = 4
    watering = 8
    health = 16
    alarms = 32
    webcam = 64
    database = 128


ManagementNames = Literal[*get_enum_names(ManagementFlags)]


@dataclass()
class ManagementConfig:
    sensors: bool = False
    light: bool = False
    climate: bool = False
    watering: bool = False
    health: bool = False
    alarms: bool = False
    webcam: bool = False
    database: bool = False

    def to_flag(self) -> int:
        flag = 0
        for management in get_enum_names(ManagementFlags):
            if getattr(self, management):
                flag += getattr(ManagementFlags, management)
        return flag


class ManagementConfigDict(TypedDict):
    sensors: bool
    light: bool
    climate: bool
    watering: bool
    health: bool
    database: bool
    alarms: bool
    webcam: bool


# Light
class ActuatorMode(Enum):
    off = "off"
    on = "on"
    automatic = "automatic"


ActuatorModeNames = Literal[*get_enum_names(ActuatorMode)]


class LightMethod(Enum):
    fixed = "fixed"
    elongate = "elongate"
    mimic = "mimic"


LightMethodNames = Literal[*get_enum_names(LightMethod)]


# Climate
@dataclass()
class ChaosConfig:
    frequency: int = 0
    duration: int = 0
    intensity: int | float = 0.0


class ChaosConfigDict(TypedDict):
    frequency: int
    duration: int
    intensity: int | float


@dataclass()
class DayConfig:
    day: time | None = time(8)
    night: time | None = time(20)

    @validator("day", "night", pre=True)
    def parse_day(cls, value: str | time | None):
        if value is None or isinstance(value, time):
            return value
        hours, minutes = value.replace('H', 'h').split("h")
        if minutes == "":
            minutes = 0
        return time(int(hours), int(minutes))


@dataclass()
class SkyConfig(DayConfig):
    lighting: LightMethod = LightMethod.fixed

    @validator("lighting", pre=True)
    def parse_lighting(cls, value):
        return safe_enum_from_name(LightMethod, value)


class SkyConfigDict(TypedDict):
    day: str
    night: str
    lighting: str


class ClimateParameter(Enum):
    temperature = "temperature"
    humidity = "humidity"
    light = "light"
    wind = "wind"


ClimateParameterNames = Literal[*get_enum_names(ClimateParameter)]


@dataclass()
class ClimateConfig:
    parameter: ClimateParameter
    day: float
    night: float
    hysteresis: float

    @validator("parameter", pre=True)
    def parse_parameter(cls, value):
        return safe_enum_from_name(ClimateParameter, value)

    @validator("day", "night", "hysteresis", pre=True)
    def cast_as_float(cls, value):
        return float(value)


class ClimateConfigDict(TypedDict):
    parameter: ClimateParameterNames
    day: int | float
    night: int | float
    hysteresis: int | float


@dataclass()
class EnvironmentConfig:
    chaos: ChaosConfig = field(default_factory=ChaosConfig)
    sky: SkyConfig = field(default_factory=SkyConfig)
    climate: list[ClimateConfig] = field(default_factory=list)

    @validator("climate", pre=True)
    def format_climate(cls, value: dict | list):
        if isinstance(value, dict):
            return [{"parameter": key, **value} for key, value in value.items()]
        return value


class EnvironmentConfigDict(TypedDict):
    chaos: ChaosConfigDict
    sky: SkyConfigDict
    climate: list[ClimateConfigDict]


class HardwareLevel(Enum):
    environment = "environment"
    plants = "plants"


HardwareLevelNames = Literal[*get_enum_names(HardwareLevel)]


class HardwareType(Enum):
    sensor = "sensor"
    light = "light"
    cooler = "cooler"
    heater = "heater"
    humidifier = "humidifier"
    dehumidifier = "dehumidifier"


HardwareTypeNames = Literal[*get_enum_names(HardwareType)]


@dataclass()
class HardwareConfig:
    uid: str
    name: str
    address: str
    type: HardwareType
    level: HardwareLevel
    model: str
    measure: list[str] = field(default_factory=list)
    plant: list[str] = field(default_factory=list)

    @validator("type", pre=True)
    def parse_type(cls, value):
        return safe_enum_from_name(HardwareType, value)

    @validator("level", pre=True)
    def parse_level(cls, value):
        return safe_enum_from_name(HardwareLevel, value)

    @validator("measure", "plant", pre=True)
    def parse_day(cls, value: str | list | None):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class HardwareConfigDict(TypedDict):
    uid: str
    name: str
    address: str
    type: str
    level: str
    model: str
    measure: list[str]
    plant: list[str]


# Data and records
@dataclass()
class MeasureRecord:
    measure: str
    value: int | float


class MeasureRecordDict(TypedDict):
    measure: str
    value: int | float


@dataclass()
class SensorRecord:
    sensor_uid: str
    measures: list[MeasureRecord] = field(default_factory=list)


class SensorRecordDict(TypedDict):
    sensor_uid: str
    measures: list[MeasureRecordDict]


@dataclass()
class SensorsData:
    timestamp: datetime
    records: list[SensorRecord] = field(default_factory=list)
    average: list[MeasureRecord] = field(default_factory=list)


class SensorsDataDict(TypedDict):
    timestamp: datetime
    records: list[SensorRecordDict]
    average: list[MeasureRecordDict]


@dataclass()
class HealthRecord:
    green: int | float
    necrosis: int | float
    index: int | float


class HealthRecordDict(TypedDict):
    green: int | float
    necrosis: int | float
    index: int | float


@dataclass()
class HealthData:
    timestamp: datetime
    data: HealthRecord


class HealthDataDict(TypedDict):
    timestamp: datetime
    data: HealthRecordDict


@dataclass()
class LightingHours:
    morning_start: time = time(8)
    morning_end: time | None = None
    evening_start: time | None = None
    evening_end: time = time(20)


class LightingHoursDict(TypedDict):
    morning_start: time
    morning_end: time | None
    evening_start: time | None
    evening_end: time


@dataclass()
class LightData(LightingHours):
    status: bool = False
    mode: ActuatorMode = ActuatorMode.automatic
    method: LightMethod = LightMethod.fixed
    timer: float = 0.0


class LightDataDict(LightingHoursDict):
    status: bool
    mode: ActuatorMode
    method: LightMethod
    timer: float


# Others
@dataclass()
class SunTimes:
    twilight_begin: time
    sunrise: time
    sunset: time
    twilight_end: time


# Broker payloads
@dataclass()
class BrokerPayload:
    uid: str
    data: Any

    @classmethod
    def from_base(cls, uid: str, base: Any):
        return cls(
            uid=uid,
            data=base,
        )


class BrokerPayloadDict(TypedDict):
    uid: str


# Config
@dataclass()
class BaseInfoConfigPayload(BrokerPayload):
    data: BaseInfoConfig


class BaseInfoConfigPayloadDict(BrokerPayloadDict):
    data: BaseInfoConfigDict


@dataclass()
class ManagementConfigPayload(BrokerPayload):
    data: ManagementConfig


class ManagementConfigPayloadDict(BrokerPayloadDict):
    data: ManagementConfigDict


@dataclass()
class EnvironmentConfigPayload(BrokerPayload):
    data: EnvironmentConfig


class EnvironmentConfigPayloadDict(BrokerPayloadDict):
    data: EnvironmentConfigDict


@dataclass()
class HardwareConfigPayload(BrokerPayload):
    data: list[HardwareConfig]


class HardwareConfigPayloadDict(BrokerPayloadDict):
    data: HardwareConfigDict


# Data
@dataclass
class SensorsDataPayload(BrokerPayload):
    data: SensorsData


class SensorsDataPayloadDict(BrokerPayloadDict):
    data: SensorsDataDict


@dataclass
class LightDataPayload(BrokerPayload):
    data: LightData


class LightDataPayloadDict(BrokerPayloadDict):
    data: LightDataDict


@dataclass
class HealthDataPayload(BrokerPayload):
    data: HealthData


class HealthDataPayloadDict(BrokerPayloadDict):
    data: HealthDataDict
