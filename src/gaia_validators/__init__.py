from __future__ import annotations

from datetime import datetime, time
from enum import Enum, IntFlag
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, validator
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
class BaseInfoConfig(BaseModel):
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


class ManagementConfig(BaseModel):
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
    automatic = "automatic"
    manual = "manual"


ActuatorModeNames = Literal[*get_enum_names(ActuatorMode)]


class ActuatorStatus(Enum):
    on = True
    off = False


ActuatorStatusNames = Literal[*get_enum_names(ActuatorStatus)]


ActuatorTurnTo = Literal["on", "off", "automatic"]


class LightMethod(Enum):
    fixed = "fixed"
    elongate = "elongate"
    mimic = "mimic"


LightMethodNames = Literal[*get_enum_names(LightMethod)]


# Climate
class ChaosConfig(BaseModel):
    frequency: int = 0
    duration: int = 0
    intensity: int | float = 0.0


class ChaosConfigDict(TypedDict):
    frequency: int
    duration: int
    intensity: int | float


class DayConfig(BaseModel):
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


class SkyConfig(DayConfig):
    lighting: LightMethod = LightMethod.fixed

    @validator("lighting", pre=True)
    def parse_lighting(cls, value):
        return safe_enum_from_name(LightMethod, value)


class SkyConfigDict(TypedDict):
    day:  time | None | str
    night:  time | None | str
    lighting: str


class ClimateParameter(Enum):
    temperature = "temperature"
    humidity = "humidity"
    light = "light"
    wind = "wind"


ClimateParameterNames = Literal[*get_enum_names(ClimateParameter)]


class ClimateConfig(BaseModel):
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
    parameter: ClimateParameter | ClimateParameterNames
    day: int | float
    night: int | float
    hysteresis: int | float


class EnvironmentConfig(BaseModel):
    chaos: ChaosConfig = Field(default_factory=ChaosConfig)
    sky: SkyConfig = Field(default_factory=SkyConfig)
    climate: list[ClimateConfig] = Field(default_factory=list)

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


class HardwareConfig(BaseModel):
    uid: str
    name: str
    address: str
    type: HardwareType
    level: HardwareLevel
    model: str
    measures: list[str] = Field(default_factory=list, alias="measure")
    plants: list[str] = Field(default_factory=list, alias="plant")

    class Config:
        allow_population_by_field_name = True

    @validator("type", pre=True)
    def parse_type(cls, value):
        return safe_enum_from_name(HardwareType, value)

    @validator("level", pre=True)
    def parse_level(cls, value):
        return safe_enum_from_name(HardwareLevel, value)

    @validator("measures", "plants", pre=True)
    def parse_to_list(cls, value: str | list | None):
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
    measures: list[str]
    plants: list[str]


# Data and records
class MeasureRecord(BaseModel):
    measure: str
    value: int | float


class MeasureRecordDict(TypedDict):
    measure: str
    value: int | float


class SensorRecord(BaseModel):
    sensor_uid: str
    measures: list[MeasureRecord] = Field(default_factory=list)


class SensorRecordDict(TypedDict):
    sensor_uid: str
    measures: list[MeasureRecordDict]


class SensorsData(BaseModel):
    timestamp: datetime
    records: list[SensorRecord] = Field(default_factory=list)
    average: list[MeasureRecord] = Field(default_factory=list)


class SensorsDataDict(TypedDict):
    timestamp: datetime | str
    records: list[SensorRecordDict]
    average: list[MeasureRecordDict]


class HealthRecord(BaseModel):
    green: int | float
    necrosis: int | float
    index: int | float


class HealthRecordDict(TypedDict):
    green: int | float
    necrosis: int | float
    index: int | float


class HealthData(HealthRecord):
    timestamp: datetime


class HealthDataDict(HealthRecordDict):
    timestamp: datetime | str


class LightingHours(BaseModel):
    morning_start: time = time(8)
    morning_end: time | None = None
    evening_start: time | None = None
    evening_end: time = time(20)


class LightingHoursDict(TypedDict):
    morning_start: time | str
    morning_end: time | None | str
    evening_start: time | None | str
    evening_end: time | str


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
class SunTimes(BaseModel):
    twilight_begin: time
    sunrise: time
    sunset: time
    twilight_end: time


# Broker payloads
class BrokerPayload(BaseModel):
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
class BaseInfoConfigPayload(BrokerPayload):
    data: BaseInfoConfig


class BaseInfoConfigPayloadDict(BrokerPayloadDict):
    data: BaseInfoConfigDict


class ManagementConfigPayload(BrokerPayload):
    data: ManagementConfig


class ManagementConfigPayloadDict(BrokerPayloadDict):
    data: ManagementConfigDict


class EnvironmentConfigPayload(BrokerPayload):
    data: EnvironmentConfig


class EnvironmentConfigPayloadDict(BrokerPayloadDict):
    data: EnvironmentConfigDict


class HardwareConfigPayload(BrokerPayload):
    data: list[HardwareConfig]


class HardwareConfigPayloadDict(BrokerPayloadDict):
    data: list[HardwareConfigDict]


# Data
class SensorsDataPayload(BrokerPayload):
    data: SensorsData


class SensorsDataPayloadDict(BrokerPayloadDict):
    data: SensorsDataDict


class LightDataPayload(BrokerPayload):
    data: LightData


class LightDataPayloadDict(BrokerPayloadDict):
    data: LightDataDict


class HealthDataPayload(BrokerPayload):
    data: HealthData


class HealthDataPayloadDict(BrokerPayloadDict):
    data: HealthDataDict
