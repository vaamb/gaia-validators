from __future__ import annotations

from datetime import datetime, time
from enum import Enum, IntFlag
from typing import Any, Literal, NamedTuple, Self, Type, TypedDict, TypeVar
from uuid import UUID, uuid4

from pydantic import (
    __version__ as pydantic_version, BaseModel as _BaseModel, Field)
from pydantic.dataclasses import dataclass

if pydantic_version >= "2.0.0":
    from pydantic import ConfigDict, field_validator

    class BaseModel(_BaseModel):
        model_config = ConfigDict(
            from_attributes=True,
            populate_by_name=True,
        )
else:
    try:
        from pydantic import field_validator
    except ImportError:
        import warnings

        warnings.warn(
            f"Pydantic version {pydantic_version} loaded. Patching it to be "
            f"compatible")

        from gaia_validators.patch import patch_pydantic_v1

        patch_pydantic_v1()

        # field_validator has been injected by patch_pydantic_v1
        from pydantic import field_validator
    finally:
        class BaseModel(_BaseModel):
            class Config:
                orm_mode = True
                allow_population_by_field_name = True

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        def __repr__(self) -> str:
            return str.__repr__(self.value)


T = TypeVar("T", bound=Enum)


def _get_enum_names(enum: Type[Enum]) -> list[str]:
    return [i.name for i in enum]


def safe_enum_from_name(enum: Type[T], name: str | T) -> T:
    """Return the enum member whose name is 'name'

    :param enum: An enum that should contain an element named 'name'
    :param name: The name of an enum element, or an element of the enum
    :return: An enum element
    """
    if isinstance(name, enum):
        if not isinstance(name, enum):
            raise ValueError(f"'{name}' does not belong to {enum}")
        return name
    try:
        return getattr(enum, name)
    except AttributeError:
        raise ValueError(f"'{name}' is not a valid {enum} name")


def safe_enum_from_value(enum: Type[T], value: str | T) -> T:
    """Return the enum member whose value is 'value'

    :param enum: An enum that should contain an element with the value 'value'
    :param value: The value of an enum element, or an element of the enum
    :return: An enum element
    """
    if isinstance(value, enum):
        if not isinstance(value, enum):
            raise ValueError(f"'{value}' does not belong to {enum}")
        return value
    try:
        return enum(value)
    except ValueError:
        raise ValueError(f"'{value}' is not a valid {enum} value")


@dataclass()
class Empty:
    pass


@dataclass(frozen=True)
class IDs:
    uid: str
    name: str

    def __iter__(self):
        return iter((self.uid, self.name))


""" Temporarily removed until a wheel is available for Raspi
class BaseModel(_BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )
"""


# Crud actions
class CrudAction(StrEnum):
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
    pictures = 64
    database = 128


ManagementNames = Literal[*_get_enum_names(ManagementFlags)]  # noqa: works when imported


class ManagementConfig(BaseModel):
    sensors: bool = False
    light: bool = False
    climate: bool = False
    watering: bool = False
    health: bool = False
    alarms: bool = False
    pictures: bool = False
    database: bool = False

    def to_flag(self) -> int:
        flag = 0
        for management in ManagementFlags:
            try:
                if getattr(self, management.name):
                    flag += management
            except AttributeError:
                pass
        return flag

    @classmethod
    def from_flag(cls, flag: int | ManagementFlags) -> Self:
        if isinstance(flag, int):
            flag = ManagementFlags(flag)
        payload = {management.name: True for management in flag}
        return cls(**payload)


class ManagementConfigDict(TypedDict):
    sensors: bool
    light: bool
    climate: bool
    watering: bool
    health: bool
    alarms: bool
    pictures: bool
    database: bool


# Actuator
class ActuatorMode(StrEnum):
    automatic = "automatic"
    manual = "manual"


ActuatorModeNames = Literal[*_get_enum_names(ActuatorMode)]  # noqa: works when imported


class ActuatorStatus(StrEnum):
    on = "on"
    off = "off"


ActuatorStatusNames = Literal[*_get_enum_names(ActuatorStatus)]  # noqa: works when imported


class ActuatorModePayload(StrEnum):
    on = "on"
    off = "off"
    automatic = "automatic"


# Light
class LightMethod(StrEnum):
    fixed = "fixed"
    elongate = "elongate"
    mimic = "mimic"


LightMethodNames = Literal[*_get_enum_names(LightMethod)]  # noqa: works when imported


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

    @field_validator("day", "night", mode="before")
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


class DayConfigDict(TypedDict):
    day: time
    night: time


class SkyConfig(DayConfig):
    lighting: LightMethod = LightMethod.fixed

    @field_validator("lighting", mode="before")
    def parse_lighting(cls, value):
        return safe_enum_from_name(LightMethod, value)


class SkyConfigDict(TypedDict):
    day:  time | None | str
    night:  time | None | str
    lighting: str


class ClimateParameter(StrEnum):
    temperature = "temperature"
    humidity = "humidity"
    light = "light"
    wind = "wind"


ClimateParameterNames = Literal[*_get_enum_names(ClimateParameter)]  # noqa: works when imported


class ClimateConfig(BaseModel):
    parameter: ClimateParameter
    day: float
    night: float
    hysteresis: float

    @field_validator("parameter", mode="before")
    def parse_parameter(cls, value):
        return safe_enum_from_name(ClimateParameter, value)

    @field_validator("day", "night", "hysteresis", mode="before")
    def cast_as_float(cls, value):
        return float(value)


class ClimateConfigDict(TypedDict):
    parameter: ClimateParameter
    day: float
    night: float
    hysteresis: float


class EnvironmentConfig(BaseModel):
    chaos: ChaosConfig = Field(default_factory=ChaosConfig)
    sky: SkyConfig = Field(default_factory=SkyConfig)
    climate: list[ClimateConfig] = Field(default_factory=list)

    @field_validator("climate", mode="before")
    def format_climate(cls, value: dict | list):
        if isinstance(value, dict):
            return [{"parameter": key, **value} for key, value in value.items()]
        return value


class EnvironmentConfigDict(TypedDict):
    chaos: ChaosConfigDict
    sky: SkyConfigDict
    climate: list[ClimateConfigDict]


# Hardware
class HardwareLevel(StrEnum):
    environment = "environment"
    plants = "plants"


HardwareLevelNames = Literal[*_get_enum_names(HardwareLevel)]  # noqa: works when imported


class HardwareType(StrEnum):
    sensor = "sensor"
    light = "light"
    cooler = "cooler"
    heater = "heater"
    humidifier = "humidifier"
    dehumidifier = "dehumidifier"


HardwareTypeNames = Literal[*_get_enum_names(HardwareType)]  # noqa: works when imported


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

    @field_validator("type", mode="before")
    def parse_type(cls, value):
        return safe_enum_from_name(HardwareType, value)

    @field_validator("level", mode="before")
    def parse_level(cls, value):
        return safe_enum_from_name(HardwareLevel, value)

    @field_validator("measures", "plants", mode="before")
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
class MeasureAverage(NamedTuple):
    measure: str
    value: float
    timestamp: datetime | None = None


class SensorRecord(NamedTuple):
    sensor_uid: str
    measure: str
    value: float
    timestamp: datetime | None = None


class SensorsData(BaseModel):
    timestamp: datetime
    records: list[SensorRecord] = Field(default_factory=list)
    average: list[MeasureAverage] = Field(default_factory=list)


class SensorsDataDict(TypedDict):
    timestamp: datetime | str
    records: list[SensorRecord]
    average: list[MeasureAverage]


class HealthRecord(NamedTuple):
    green: float
    necrosis: float
    index: float
    timestamp: datetime | None


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

    @field_validator("mode", mode="before")
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
    data: HealthRecord


class HealthDataPayloadDict(EcosystemPayloadDict):
    data: HealthRecord


# Actuators payload
class TurnActuatorPayload(BaseModel):
    ecosystem_uid: str | None = None  # can be None if transferred in parallel
    actuator: HardwareType
    mode: ActuatorModePayload = ActuatorModePayload.automatic
    countdown: float = 0.0

    @field_validator("actuator", mode="before")
    def parse_actuator(cls, value):
        return safe_enum_from_name(HardwareType, value)

    @field_validator("mode", mode="before")
    def parse_mode(cls, value):
        return safe_enum_from_name(ActuatorModePayload, value)


class TurnActuatorPayloadDict(TypedDict):
    ecosystem_uid: str | None
    actuator: HardwareType
    mode: ActuatorModePayload
    countdown: float


class Route(BaseModel):
    engine_uid: str
    ecosystem_uid: str | None = None


class RouteDict(TypedDict):
    engine_uid: str
    ecosystem_uid: str | None


# Crud payload
class CrudPayload(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    routing: Route
    action: CrudAction
    target: str
    data: str | dict = Field(default_factory=dict)

    @field_validator("action", mode="before")
    def parse_action(cls, value):
        return safe_enum_from_name(CrudAction, value)


class CrudPayloadDict(TypedDict):
    uuid: str
    routing: RouteDict
    action: CrudAction
    target: str
    data: str | dict


# Buffered data payloads
class BufferedSensorRecord(NamedTuple):
    ecosystem_uid: str
    sensor_uid: str
    measure: str
    value: float
    timestamp: datetime


class BufferedSensorsDataPayload(BaseModel):
    data: list[BufferedSensorRecord]
    uuid: UUID


class BufferedSensorsDataPayloadDict(TypedDict):
    data: list[BufferedSensorRecord]
    uuid: UUID


# Request (CRUD & buffered data saving) results
class Result(StrEnum):
    success = "success"
    failure = "failure"


class RequestResult(BaseModel):
    uuid: UUID
    status: Result
    message: str | None = None


class RequestResultDict(TypedDict):
    uuid: str
    status: Result
    message: str | None


_imported = {
    "_BaseModel", "annotations", "Any", "dataclass", "datetime", "EnumType",
    "Field", "field_validator", "IntFlag", "Literal", "StrEnum", "time",
    "TypedDict", "UUID", "uuid4"
}

__all__ = [_ for _ in dir() if _ not in ["_imported", *_imported, *__builtins__]]
