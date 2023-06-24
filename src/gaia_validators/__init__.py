from __future__ import annotations

from datetime import datetime, time
from enum import Enum, EnumType, IntFlag
from typing import Any, Literal, TypedDict

from pydantic import BaseModel as _BaseModel, Field, validator
from pydantic.dataclasses import dataclass


def get_enum_names(enum: EnumType) -> list:
    return [i.name for i in enum]


def safe_enum_from_name(enum: EnumType, name: str | Enum) -> Enum:
    if isinstance(name, str):
        return {i.name: i for i in enum}[name]
    return name


def safe_enum_from_value(enum: EnumType, value: str | Enum) -> Enum:
    if isinstance(value, str):
        return {i.value: i for i in enum}[value]
    return value


@dataclass()
class Empty:
    pass


@dataclass(frozen=True)
class IDs:
    uid: str
    name: str

    def __iter__(self):
        return iter((self.uid, self.name))


class BaseModel(_BaseModel):
    class Config:
        orm_mode = True


# Crud actions
class CrudAction(Enum):
    create = "create"
    get = "get"  # Don't like "read"
    update = "update"
    delete = "delete"


# Config
class BaseInfoConfig(BaseModel):
    engine_uid: str
    uid: str
    name: str
    status: bool = False


class BaseInfoConfigDict(TypedDict):
    engine_uid: str
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


# Actuator
class ActuatorMode(Enum):
    automatic = "automatic"
    manual = "manual"


ActuatorModeNames = Literal[*get_enum_names(ActuatorMode)]


class ActuatorStatus(Enum):
    on = True
    off = False


ActuatorStatusNames = Literal[*get_enum_names(ActuatorStatus)]


class ActuatorModePayload(Enum):
    on = "on"
    off = "off"
    automatic = "automatic"


# Light
class LightMethod(Enum):
    fixed = "fixed"
    elongate = "elongate"
    mimic = "mimic"


LightMethodNames = Literal[*get_enum_names(LightMethod)]


# Climate
class ChaosConfig(BaseModel):
    frequency: int = 0
    duration: int = 0
    intensity: float = 0.0


class ChaosConfigDict(TypedDict):
    frequency: int
    duration: int
    intensity: float


class DayConfig(BaseModel):
    day: time | None = time(8)
    night: time | None = time(20)

    @validator("day", "night", pre=True)
    def parse_day(cls, value: str | time | None):
        if value is None or isinstance(value, time):
            return value
        try:
            return time.fromisoformat(value)
        except ValueError:            
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
    day: float
    night: float
    hysteresis: float


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


# Hardware
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
    multiplexer_model: str | None = Field(default=None, alias="multiplexer")

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
    multiplexer_model: str | None


# Data and records
class MeasureRecord(BaseModel):
    measure: str
    value: float


class MeasureRecordDict(TypedDict):
    measure: str
    value: float


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
    green: float
    necrosis: float
    index: float


class HealthRecordDict(TypedDict):
    green: float
    necrosis: float
    index: float


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
    method: LightMethod = LightMethod.fixed


class LightDataDict(LightingHoursDict):
    method: LightMethod


# Actuators data
class ActuatorState(BaseModel):
    active: bool = False
    status: bool = False
    mode: ActuatorMode = ActuatorMode.automatic

    @validator("mode", pre=True)
    def parse_mode(cls, value):
        return safe_enum_from_name(ActuatorMode, value)


class ActuatorStateDict(TypedDict):
    active: bool
    status: bool
    mode: ActuatorMode


class ActuatorsData(BaseModel):
    light: ActuatorState = ActuatorState()
    cooler: ActuatorState = ActuatorState()
    heater: ActuatorState = ActuatorState()
    humidifier: ActuatorState = ActuatorState()
    dehumidifier: ActuatorState = ActuatorState()


class ActuatorsDataDict(TypedDict):
    light: ActuatorStateDict
    cooler: ActuatorStateDict
    heater: ActuatorStateDict
    humidifier: ActuatorStateDict
    dehumidifier: ActuatorStateDict


# Others
class SunTimes(BaseModel):
    twilight_begin: time
    sunrise: time
    sunset: time
    twilight_end: time


# Broker payloads
class EnginePayload(BaseModel):
    engine_uid: str
    address: str


class EnginePayloadDict(TypedDict):
    engine_uid: str
    address: str


class EcosystemPayload(BaseModel):
    uid: str
    data: Any

    @classmethod
    def from_base(cls, uid: str, base: Any):
        return cls(
            uid=uid,
            data=base,
        )


class EcosystemPayloadDict(TypedDict):
    uid: str


# Config Payload
class BaseInfoConfigPayload(EcosystemPayload):
    data: BaseInfoConfig


class BaseInfoConfigPayloadDict(EcosystemPayloadDict):
    data: BaseInfoConfigDict


class ManagementConfigPayload(EcosystemPayload):
    data: ManagementConfig


class ManagementConfigPayloadDict(EcosystemPayloadDict):
    data: ManagementConfigDict


class EnvironmentConfigPayload(EcosystemPayload):
    data: EnvironmentConfig


class EnvironmentConfigPayloadDict(EcosystemPayloadDict):
    data: EnvironmentConfigDict


class HardwareConfigPayload(EcosystemPayload):
    data: list[HardwareConfig]


class HardwareConfigPayloadDict(EcosystemPayloadDict):
    data: list[HardwareConfigDict]


# Data payloads
class SensorsDataPayload(EcosystemPayload):
    data: SensorsData


class SensorsDataPayloadDict(EcosystemPayloadDict):
    data: SensorsDataDict


class LightDataPayload(EcosystemPayload):
    data: LightData


class LightDataPayloadDict(EcosystemPayloadDict):
    data: LightDataDict


class ActuatorsDataPayload(EcosystemPayload):
    data: ActuatorsData


class ActuatorsDataPayloadDict(EcosystemPayloadDict):
    data: ActuatorsDataDict


class HealthDataPayload(EcosystemPayload):
    data: HealthData


class HealthDataPayloadDict(EcosystemPayloadDict):
    data: HealthDataDict


# Actuators payload
class TurnActuatorPayload(BaseModel):
    ecosystem_uid: str | None = None  # can be None if transferred in parallel
    actuator: HardwareType
    mode: ActuatorModePayload = ActuatorModePayload.automatic
    countdown: float = 0.0

    @validator("actuator", pre=True)
    def parse_actuator(cls, value):
        return safe_enum_from_name(HardwareType, value)

    @validator("mode", pre=True)
    def parse_mode(cls, value):
        return safe_enum_from_name(ActuatorModePayload, value)


class TurnActuatorPayloadDict(TypedDict):
    ecosystem_uid: str | None
    actuator: HardwareType
    mode: ActuatorModePayload
    countdown: float


# Crud payload
class CrudPayload(BaseModel):
    engine_uid: str
    action: CrudAction
    target: str
    values: dict = Field(default_factory=dict)

    @validator("action", pre=True)
    def parse_actuator(cls, value):
        return safe_enum_from_name(CrudAction, value)


class CrudPayloadDict(TypedDict):
    engine_uid: str
    action: CrudAction
    target: str
    values: dict
