from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from enum import auto, Enum, IntEnum, IntFlag, StrEnum
import sys
from typing import (
    Any,
    Generic,
    ItemsView,
    Iterable,
    NamedTuple,
    NotRequired,
    overload,
    Self,
    Sequence,
    Type,
    TypedDict,
    TypeVar,
)
from uuid import UUID, uuid4

from pydantic import (
    BaseModel as _BaseModel, ConfigDict, Field, field_serializer, field_validator,
    GetCoreSchemaHandler, model_serializer, model_validator,
    SerializerFunctionWrapHandler, SerializationInfo)
from pydantic.dataclasses import dataclass
from pydantic_core import core_schema, PydanticUseDefault


if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs


T = TypeVar("T", bound=Enum)
DictT = TypeVar("DictT")


def safe_enum_from_name(enum: Type[T], name: str | T) -> T:
    """Return the enum member whose name is 'name'

    :param enum: An enum that should contain an element named 'name'
    :param name: The name of an enum element, or an element of the enum
    :return: An enum element
    """
    if isinstance(name, enum):
        return name
    try:
        return getattr(enum, name)  # type: ignore
    except AttributeError:
        raise ValueError(f"'{name}' is not a valid {enum} name")


def safe_enum_from_value(enum: Type[T], value: str | T) -> T:
    """Return the enum member whose value is 'value'

    :param enum: An enum that should contain an element with the value 'value'
    :param value: The value of an enum element, or an element of the enum
    :return: An enum element
    """
    try:
        return enum(value)
    except ValueError:
        raise ValueError(f"'{value}' is not a valid {enum} value")


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class LaxBaseModel(_BaseModel, Generic[DictT]):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )

    def model_dump(self, **kwargs) -> DictT:
        return super().model_dump(**kwargs)  # type: ignore[return-value]


class BaseModel(_BaseModel, Generic[DictT]):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        populate_by_name=True,
    )

    def model_dump(self, **kwargs) -> DictT:
        return super().model_dump(**kwargs)  # type: ignore[return-value]


class Empty:
    """A class for empty data/payload.

    Used by Gaia.
    """
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance


empty = Empty()


def is_empty(value: Any) -> TypeIs[Empty]:
    return isinstance(value, Empty)


class MissingValue:
    """A sentinel class to mark for missing value in update payloads

    Used by Ouranos.
    """
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<MISSING_VALUE>"

    @classmethod
    def __get_pydantic_core_schema__(
            cls,
            source_type: Any,
            _: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(
            cls=source_type,
            serialization=core_schema.to_string_ser_schema(),
        )


missing = MissingValue()


def is_missing(value: Any) -> TypeIs[MissingValue]:
    return isinstance(value, MissingValue)


@dataclass(frozen=True)
class IDs:
    """A Gaia ecosystem id

    Gaia ecosystems have a name, that can be shared between multiple Gaia
    instances in the case of an ecosystem managed by multiple Raspberry Pi's
    and a unique uid.

    Used by Gaia and Ouranos.
    """
    uid: str
    name: str

    def __iter__(self):
        return iter((self.uid, self.name))


EcosystemIDs = IDs


class PeriodOfDay(Enum):
    day = auto()
    night = auto()


# Crud actions
class CrudAction(StrEnum):
    """Possible crud actions.

    Used by Gaia and Ouranos.
    """
    create = "create"
    get = "get"  # Don't like "read"
    update = "update"
    delete = "delete"


# Warnings
class WarningLevel(IntEnum):
    low = 0
    moderate = 1
    high = 2
    severe = 3
    critical = 4


class Position(Enum):
    under = -1
    none = 0
    above = 1


# Config
class BaseInfoConfigDict(TypedDict):
    """Cf. related BaseModel."""
    engine_uid: str
    uid: str
    name: str
    status: bool


class BaseInfoConfig(BaseModel[BaseInfoConfigDict]):
    """Minimum info about a Gaia ecosystem needed for Ouranos to register it.

    Used by Gaia and Ouranos.
    """
    engine_uid: str
    uid: str
    name: str
    status: bool = False


# Management
class ManagementFlags(IntFlag):
    """Short form of a Gaia ecosystem subroutine management info.

    Used by Ouranos to save the info in the database while allowing to add new
    subroutines in the future.
    """
    # Basal managements
    sensors = 1
    light = 2
    camera = 4
    database = 8
    weather = 16
    # available basal management = 32
    # available basal management = 64
    # available basal management = 128

    # Managements with dependencies
    alarms = 256
    climate = 512
    watering = 1024
    health = 2048
    pictures = 4096

    # Checks for managements with dependencies
    alarms_enabled = sensors | alarms
    climate_enabled = sensors | climate
    watering_enabled = sensors | watering
    health_enabled = camera | health
    pictures_enabled = camera | pictures


class ManagementConfigDict(TypedDict):
    """Cf. related BaseModel."""
    sensors: bool
    light: bool
    camera: bool
    database: bool
    weather: bool

    alarms: bool
    climate: bool
    watering: bool
    health: bool
    pictures: bool


class ManagementConfig(BaseModel[ManagementConfigDict]):
    """Long form of a Gaia ecosystem subroutine management info.

    Used by Gaia in the ecosystems configuration file.
    """
    sensors: bool = False
    light: bool = False
    camera: bool = False
    database: bool = False
    weather: bool = False

    alarms: bool = False
    climate: bool = False
    watering: bool = False
    health: bool = False
    pictures: bool = False

    def to_flag(self) -> int:
        flag = 0
        for management, value in self.__dict__.items():
            if not value:
                continue
            try:
                flag += safe_enum_from_name(ManagementFlags, management)
            except ValueError:
                pass
        return flag

    @classmethod
    def from_flag(cls, flag: int | ManagementFlags) -> Self:
        if isinstance(flag, int):
            flag = ManagementFlags(flag)
        payload = {management.name: True for management in flag}
        return cls(**payload)  # type: ignore


# Actuator
class ActuatorMode(StrEnum):
    """Actuator management mode.

    If automatic, the actuator is managed by one or more ecosystem subroutine.
    If manual, the actuator status has been overriden manually (via a request
    coming from Ouranos).

    Used by Gaia and Ouranos.
    """
    automatic = "automatic"
    manual = "manual"


class ActuatorStatus(StrEnum):
    """Actuator status.

    Used by Gaia and Ouranos.
    """
    on = "on"
    off = "off"


class ActuatorModePayload(StrEnum):
    """Instruction on how to change actuator state.

    If 'on' or 'off' is used, the actuator mode will be changed to 'manual' and
    its status will be set to 'on' or 'off'.
    If 'automatic' is used, the actuator mode will be changed to 'automatic' and
    its status will be managed by one or more ecosystem subroutines.

    Used by Gaia and Ouranos.
    """
    on = "on"
    off = "off"
    automatic = "automatic"
    auto = automatic


# Light
class NycthemeralSpanMethod(IntFlag):
    """Lighting hours span.

    If 'fixed', 'LightingHours.morning_start' and 'LightingHours.evening_end'
    will be set to 'DayConfig.day' and 'DayConfig.night', respectively.
    If 'mimic', 'LightingHours.morning_start' and 'LightingHours.evening_end'
    will be computed based on the sunrise and sunset times of a place specified
    by 'environment.nycthemeral_cycle.target'

    Used by Gaia and Ouranos.
    """
    fixed = 0
    mimic = 2


class LightingMethod(IntFlag):
    """Lighting method.

    If 'fixed', lights will be on between 'LightingHours.morning_start' and
    'LightingHours.evening_end'.
    If 'elongate', lights will be on between 'LightingHours.morning_start' and
    'LightingHours.morning_end', and 'LightingHours.evening_start' and
    'LightingHours.evening_end' (if possible).

    Used by Gaia and Ouranos.
    """
    fixed = 0
    elongate = 1


LightMethod = LightingMethod


# Climate
class ChaosConfigDict(TypedDict):
    """Cf. related BaseModel."""
    frequency: int
    duration: int
    intensity: float


class ChaosConfig(BaseModel[ChaosConfigDict]):
    """Chaos parameters.

    :arg frequency: the average delay between two chaotic events. If set to 0,
                     no chaotic event will occur.
    :arg duration: the duration in day of one chaotic event.
    :arg Intensity: the intensity of a chaotic event. It influences the
                     temperature, humidity, light and watering levels.

    Used by Gaia ecosystems configuration file.
    """
    frequency: int = Field(default=0, ge=0)
    duration: int = Field(default=0, ge=0)
    intensity: float = Field(default=0.0, ge=0.0)


class TimeWindowDict(TypedDict):
    beginning: datetime | None
    end: datetime | None


class TimeWindow(BaseModel[TimeWindowDict]):
    beginning: datetime | None = None
    end: datetime | None = None

    @field_validator("beginning", "end", mode="before")
    @classmethod
    def parse_time(cls, value):
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
            return dt.astimezone(timezone.utc)
        return value


class ChaosParametersDict(ChaosConfigDict):
    time_window: TimeWindow | TimeWindowDict


class ChaosParameters(BaseModel[ChaosParametersDict], ChaosConfig):
    time_window: TimeWindow = Field(default_factory=TimeWindow)


class NycthemeralSpanConfigDict(TypedDict):
    """Cf. related BaseModel."""
    day: time
    night: time


class NycthemeralSpanConfig(BaseModel[NycthemeralSpanConfigDict]):
    """Info about the day and night times.

    Used by Gaia ecosystems configuration file.

    Rem: if the environment light method used is `LightMethod.elongate` or
    `LightMethod.mimic`, the times given will be overriden.
    """
    day: time = time(8)
    night: time = time(20)

    @field_validator("day", "night", mode="before")
    @classmethod
    def parse_day(cls, value: str | time | None):
        if value is None or isinstance(value, time):
            return value
        try:
            return time.fromisoformat(value)
        except ValueError:
            if "h" not in value.lower():
                raise ValueError(
                    "Wrong time format. It should either be written in a valid"
                    "ISO format or as 'hours'H'minutes'"
                )
            hours, minutes = value.replace('H', 'h').split("h")
            if minutes == "":
                minutes = "0"
            return time(int(hours), int(minutes))


class NycthemeralCycleConfigDict(TypedDict):
    """Cf. related BaseModel."""
    span: NycthemeralSpanMethod
    lighting: LightingMethod
    target: str | None
    day:  time
    night:  time


SkyConfigDict = NycthemeralCycleConfigDict


class NycthemeralCycleConfig(BaseModel[NycthemeralCycleConfigDict], NycthemeralSpanConfig):
    """An augmented version of `DayConfig` with the light method added.

    Used by Gaia ecosystems configuration file.
    """
    span: NycthemeralSpanMethod = NycthemeralSpanMethod.fixed
    lighting: LightingMethod = LightingMethod.fixed
    target: str | None = None

    @field_validator("span", mode="before")
    @classmethod
    def parse_span(cls, value):
        if isinstance(value, int):
            return NycthemeralSpanMethod(value)
        return safe_enum_from_name(NycthemeralSpanMethod, value)

    @field_validator("lighting", mode="before")
    @classmethod
    def parse_lighting(cls, value):
        if isinstance(value, int):
            return LightingMethod(value)
        return safe_enum_from_name(LightingMethod, value)


SkyConfig = NycthemeralCycleConfig


class ClimateParameter(StrEnum):
    """Climate parameters that can be controlled by Gaia ecosystem subroutines.

    Climate parameters allow to control the environment of a Gaia ecosystem on
    the long-term scale.

    Used by Gaia and Ouranos.
    """
    temperature = "temperature"
    humidity = "humidity"
    light = "light"
    wind = "wind"


class ActuatorCoupleDict(TypedDict):
    increase: str | None
    decrease: str | None


class ActuatorCouple(BaseModel[ActuatorCoupleDict]):
    increase: str | None
    decrease: str | None

    def __iter__(self) -> Iterable[str | None]:
        return iter((self.increase, self.decrease))

    def items(self) -> ItemsView[str, str | None]:
        return self.__dict__.items()

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: Any):
        if "couple" in data:
            if len(data["couple"]) != 2:
                raise ValueError("ActuatorCouple's couple should be a tuple of length 2")
            return {
                "increase": data["couple"][0],
                "decrease": data["couple"][1],
            }
        return data

    @field_validator("increase", "decrease", mode="before")
    @classmethod
    def validate_increase_decrease(cls, value):
        if isinstance(value, Enum):
            return value.name
        return value


class AnonymousClimateConfigDict(TypedDict):
    """Cf. related BaseModel."""
    day: float
    night: float
    hysteresis: NotRequired[float]
    alarm: NotRequired[float | None]
    linked_actuators: NotRequired[ActuatorCoupleDict | None]
    linked_measure: NotRequired[str | None]


class AnonymousClimateConfig(BaseModel[AnonymousClimateConfigDict]):
    """Configuration for controlling one climatic parameter.

    Used by Gaia in the ecosystems configuration file.

    Rem: AnonymousClimateConfig does not store the parameter name and should
    only be used when its named is linked to the config (for ex: in a dict)
    """
    day: float
    night: float
    hysteresis: float = 0.0
    alarm: float | None = None
    linked_actuators: ActuatorCouple | None = None
    linked_measure: str | None = None

    @field_validator("day", "night", "hysteresis", mode="before")
    @classmethod
    def cast_as_float(cls, value):
        return float(value)

    @field_validator("alarm", mode="before")
    @classmethod
    def cast_as_float_or_none(cls, value):
        if value is None:
            return None
        return float(value)


class ClimateConfigDict(AnonymousClimateConfigDict):
    """Cf. related BaseModel."""
    parameter: ClimateParameter


class ClimateConfig(BaseModel[ClimateConfigDict], AnonymousClimateConfig):
    """Configuration info for a single climate parameter.

    Used by Ouranos.
    """
    parameter: ClimateParameter

    @field_validator("parameter", mode="before")
    @classmethod
    def parse_parameter(cls, value):
        return safe_enum_from_name(ClimateParameter, value)


class WeatherParameter(StrEnum):
    """Weather parameters that can be controlled by Gaia ecosystem subroutines.

    Weather parameters allow to control the environment of a Gaia ecosystem on
    the short-term scale. They are typically used to simulate short-term
    disturbing events.

    Used by Gaia and Ouranos.
    """
    rain = "rain"
    fog = "fog"
    wind_gust = "wind_gust"


class AnonymousWeatherConfigDict(TypedDict):
    """Cf. related BaseModel."""
    pattern: str
    duration: float
    level: NotRequired[float]
    linked_actuator: NotRequired[str | None]


class AnonymousWeatherConfig(BaseModel[AnonymousWeatherConfigDict]):
    """Configuration for controlling one weather parameter.

    Used by Gaia in the ecosystems configuration file.

    Rem: AnonymousWeatherConfig does not store the parameter name and should
    only be used when its named is linked to the config (for ex: in a dict)

    pattern is a Cron-like string
    """
    pattern: str = Field(pattern=r"^(((\d+,)+\d+|(\d+(\/|-)\d+)|\d+|\*(\/\d+)?) ?){5}$")
    duration: float = Field(gt=0.0)
    level: float = Field(default=100.0, ge=0.0, le=100.0)
    linked_actuator: str | None = None


class WeatherConfigDict(AnonymousWeatherConfigDict):
    """Cf. related BaseModel."""
    parameter: WeatherParameter


class WeatherConfig(BaseModel[WeatherConfigDict], AnonymousWeatherConfig):
    """Configuration info for a single weather parameter.

    Used by Ouranos.
    """
    parameter: WeatherParameter

    @field_validator("parameter", mode="before")
    @classmethod
    def parse_parameter(cls, value):
        return safe_enum_from_name(WeatherParameter, value)


# Hardware
class HardwareLevel(StrEnum):
    """Level at which the hardware operates."""
    ecosystem = "ecosystem"
    environment = "environment"
    plants = "plants"


class HardwareType(IntFlag):
    """Types of hardware possible"""
    sensor = auto()
    light = auto()
    heater = auto()
    cooler = auto()
    humidifier = auto()
    dehumidifier = auto()
    fan = auto()
    camera = auto()
    actuator = light | heater | cooler | humidifier | dehumidifier | fan
    temperature_actuator = heater | cooler
    humidity_actuator = humidifier | dehumidifier
    climate_actuator = temperature_actuator | humidity_actuator


class MeasureDict(TypedDict):
    name: str
    unit: str | None


class Measure(BaseModel[MeasureDict]):
    name: str
    unit: str | None = None


class SerializableMeasure(Measure):
    @model_serializer
    def serialize_model(self) -> str:
        return f"{self.name}|{self.unit if self.unit is not None else ''}"


class AnonymousHardwareConfigDict(TypedDict):
    """Cf. related BaseModel."""
    name: str
    active: NotRequired[bool]
    address: str
    type: HardwareType
    level: HardwareLevel
    groups: NotRequired[list[str]]
    model: str
    measures: NotRequired[list[MeasureDict]]
    plants: NotRequired[list[str]]
    multiplexer_model: NotRequired[str | None]


class AnonymousHardwareConfig(BaseModel[AnonymousHardwareConfigDict]):
    """Configuration info for a single unidentified piece of hardware.

    Used by Gaia in the ecosystems configuration file.

    Rem: HardwareConfig does not store the hardware uid and should only be used
    when its id is linked to the config (for ex: in a dict)
    """
    name: str
    active: bool = True
    address: str
    type: HardwareType
    level: HardwareLevel
    groups: list[str] = Field(default_factory=lambda data: [data["type"].name])
    model: str
    measures: list[Measure] = Field(default_factory=list, validation_alias="measure")
    plants: list[str] = Field(default_factory=list, validation_alias="plant")
    multiplexer_model: str | None = Field(default=None, validation_alias="multiplexer")

    @field_validator("type", mode="before")
    @classmethod
    def parse_type(cls, value):
        if isinstance(value, int):
            return HardwareType(value)
        return safe_enum_from_name(HardwareType, value)

    @field_validator("level", mode="before")
    @classmethod
    def parse_level(cls, value):
        return safe_enum_from_name(HardwareLevel, value)

    @field_validator("groups", mode="before")
    @classmethod
    def parse_groups(cls, value: str | list[str]):
        if value is None:
            raise PydanticUseDefault
        if isinstance(value, str):
            return [value]
        elif isinstance(value, (Sequence, set)):
            rv = [*{*value}]
            rv.sort()
            return rv
        else:
            raise ValueError(f"Value of type {type(value)} is not supported")

    @field_serializer("groups")
    def serialize_groups(self, value: list[str]):
        rv = [*{*value}]
        rv.sort()
        return rv

    @field_validator("measures", mode="before")
    @classmethod
    def parse_measures(cls, value: str | list[str] | list[dict[str, str | None]] | None):
        if value is None:
            raise PydanticUseDefault
        if isinstance(value, str):
            value = [value]
        rv = []
        for v in value:
            if isinstance(v, str):
                v_split = v.split("|")
                name = v_split[0]
                unit = v_split[1] if len(v_split) > 1 else None
                rv.append({"name": name, "unit": unit})
            else:
                rv.append(v)
        return rv

    @field_validator("plants", mode="before")
    @classmethod
    def parse_plants(cls, value: str | list[str] | None):
        if value is None:
            raise PydanticUseDefault
        if isinstance(value, str):
            return [value]
        elif isinstance(value, (Sequence, set)):
            rv = [*{*value}]
            rv.sort()
            return rv
        else:
            raise ValueError(f"Value of type {type(value)} is not supported")

    @model_serializer(mode="wrap")
    def serialize_model(
            self,
            handler: SerializerFunctionWrapHandler,
            info: SerializationInfo,
    ) -> dict[str, object]:
        serialized  = handler(self)
        if info.exclude_defaults and self.groups == [self.type.name]:
            serialized.pop("groups")
        return serialized


class HardwareConfigDict(AnonymousHardwareConfigDict):
    """Cf. related BaseModel."""
    uid: str


class HardwareConfig(BaseModel[HardwareConfigDict], AnonymousHardwareConfig):
    """Configuration info for a single piece of hardware.

    Used by Ouranos.
    """
    uid: str


# Plants
class AnonymousPlantConfigDict(TypedDict):
    name: str
    species: NotRequired[str | None]
    sowing_date: NotRequired[datetime | None]
    hardware: NotRequired[list[str]]


class AnonymousPlantConfig(BaseModel[AnonymousPlantConfigDict]):
    name: str
    species: str | None = None
    sowing_date: datetime | None = None
    hardware: list[str] = Field(default_factory=list)

    @field_validator("sowing_date", mode="before")
    @classmethod
    def parse_sowing_date(cls, value: str | datetime | None):
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    @field_validator("hardware", mode="before")
    @classmethod
    def parse_hardware(cls, value: str | list[str] | None):
        if value is None:
            raise PydanticUseDefault
        if isinstance(value, str):
            return [value]
        return value


class PlantConfigDict(AnonymousPlantConfigDict):
    uid: str


class PlantConfig(BaseModel[PlantConfigDict], AnonymousPlantConfig):
    uid: str


# Data, records and warnings
class MeasureAverage(NamedTuple):
    """Averaged sensor data for a measure

    :arg measure: the name of the measure taken.
    :arg value: the average value of the measurement taken by all the sensors.
    :arg timestamp: the timestamp of when the measurement was done. If None,
                     the timestamp is the one given by the 'SensorsData'
                     containing this 'MeasureAverage'

    Used by Gaia sensors subroutine and as part of a payload sent between Gaia
    and Ouranos.
    """
    measure: str
    value: float
    timestamp: datetime | None = None


class SensorRecord(NamedTuple):
    """Sensor data for single measurement

    :arg sensor_uid: the uid of the sensor that took the measurement.
    :arg measure: the name of the measure taken.
    :arg value: the value of the sensor's measurement.
    :arg timestamp: the timestamp of when the measurement was done. If None,
                     the timestamp is the one given by the 'SensorsData'
                     containing this 'SensorRecord'

    Used by Gaia sensors subroutine and as part of a payload sent between Gaia
    and Ouranos.
    """
    sensor_uid: str
    measure: str
    value: float
    timestamp: datetime | None = None


class SensorAlarm(NamedTuple):
    """A warning created by Gaia when an ecosystem encounters an issue.

    Used by Gaia and Ouranos.
    """
    sensor_uid: str
    measure: str
    position: Position
    delta: float
    level: WarningLevel


class SensorsDataDict(TypedDict):
    """Cf. related BaseModel."""
    timestamp: datetime | str
    records: list[SensorRecord]
    average: list[MeasureAverage]
    alarms: list[SensorAlarm]


class SensorsData(BaseModel[SensorsDataDict]):
    """A collection of all the sensor measurements for one ecosystem at one time
    point.

    For the detail of the various attributes, see the related models.

    Used by Gaia sensors subroutine and as part of a payload sent between Gaia
    and Ouranos."""
    timestamp: datetime = Field(default_factory=_now)
    records: list[SensorRecord] = Field(default_factory=list)
    average: list[MeasureAverage] = Field(default_factory=list)
    alarms: list[SensorAlarm] = Field(default_factory=list)


HealthRecord = SensorRecord


class HealthDataDict(TypedDict):
    """Cf. related BaseModel."""
    timestamp: datetime | str
    records: list[HealthRecord]


class HealthData(BaseModel[HealthDataDict]):
    """A collection of all the health measurements for one ecosystem at one time
    point.

    For the detail of the various attributes, see the related models.

    Used by Gaia sensors subroutine and as part of a payload sent between Gaia
    and Ouranos."""
    timestamp: datetime = Field(default_factory=_now)
    records: list[HealthRecord] = Field(default_factory=list)


class LightingHoursDict(TypedDict):
    """Cf. related BaseModel."""
    morning_start: time | str
    morning_end: time | None | str
    evening_start: time | None | str
    evening_end: time | str


class LightingHours(BaseModel[LightingHoursDict]):
    """Information about the lighting hours.

    :arg morning_start: time at which the lighting starts in the morning.
    :arg morning_end: time at which the lighting stops in the morning. If None,
                       evening_start should also be None and lighting will
                       be from 'morning_start' until 'evening_end'.
    :arg evening_start: time at which the lighting starts in the
                         evening/afternoon. If None, morning_end should also be
                         None and lighting will be from 'morning_start' until
                         'evening_end'.
    :arg evening_end: time at which the lighting stops in the evening/afternoon.

    Used by Gaia.

    Rem: Lighting hours depend on 'LightMethod', 'DayConfig' and sunrise and
    sunset data available.
    """
    morning_start: time = time(8)
    morning_end: time | None = None
    evening_start: time | None = None
    evening_end: time = time(20)


class LightDataDict(LightingHoursDict):
    """Cf. related BaseModel."""
    method: LightMethod


class LightData(BaseModel[LightDataDict], LightingHours):
    """An augmented version of `LightingHours` with the light method added.

    Used by Gaia.
    """
    method: LightMethod = LightMethod.fixed

    @classmethod
    def from_lighting_hours(
            cls,
            lighting_hours: LightingHours,
            method: LightMethod
    ) -> Self:
        return cls(
            morning_start=lighting_hours.morning_start,
            morning_end=lighting_hours.morning_end,
            evening_start=lighting_hours.evening_start,
            evening_end=lighting_hours.evening_end,
            method=method,
        )


class NycthemeralCycleInfoDict(NycthemeralCycleConfigDict, LightingHoursDict):
    """Cf. related BaseModel."""


class NycthemeralCycleInfo(BaseModel[NycthemeralCycleInfoDict], LightingHours, NycthemeralCycleConfig):
    """An augmented version of `NycthemeralCycleConfig` with the `LightingHours`
    added.

    Used by Gaia.
    """


# Actuators data
class ActuatorStateDict(TypedDict):
    """Cf. related BaseModel."""
    active: bool
    status: bool
    level: float | None
    mode: ActuatorMode


class ActuatorState(BaseModel[ActuatorStateDict]):
    """The state of one (type of) actuator.

    Used by Gaia and Ouranos api.
    """
    active: bool = False
    status: bool = False
    level: float | None = None
    mode: ActuatorMode = ActuatorMode.automatic

    @field_validator("mode", mode="before")
    @classmethod
    def parse_mode(cls, value):
        return safe_enum_from_name(ActuatorMode, value)


class ActuatorStateRecord(NamedTuple):
    """Actuator state at a given time.

    :arg type: the hardware type that changed state.
    :arg active: the new actuator activity status.
    :arg mode: the new actuator mode.
    :arg status: the new actuator status.
    :arg timestamp: the timestamp when the status changed.

    Used by Gaia events and as part of a payload sent between Gaia and Ouranos.
    """
    type: HardwareType
    group: str
    active: bool
    mode: ActuatorMode
    status: bool
    level: float | None
    timestamp: datetime


# Places
class Coordinates(NamedTuple):
    latitude: float
    longitude: float


class PlaceDict(TypedDict):
    name: str
    coordinates: Coordinates


class Place(BaseModel[PlaceDict]):
    name: str
    coordinates: Coordinates


# Sun times
class SunTimesDict(TypedDict):
    """Cf. related BaseModel."""
    datestamp: date | None
    astronomical_dawn: time | None
    nautical_dawn: time | None
    civil_dawn: time | None
    sunrise: time | None
    solar_noon: time | None
    sunset: time | None
    civil_dusk: time | None
    nautical_dusk: time | None
    astronomical_dusk: time | None


class SunTimes(LaxBaseModel[SunTimesDict]):
    """Information about sunrise and sunset events for a given place.

    Used by Gaia and Ouranos.
    """
    datestamp: date | None = None
    astronomical_dawn: time | None = None
    nautical_dawn: time | None = None
    civil_dawn: time | None = None
    sunrise: time | None = None
    solar_noon: time | None = None
    sunset: time | None = None
    civil_dusk: time | None = None
    nautical_dusk: time | None = None
    astronomical_dusk: time | None = None

    @property
    def twilight_duration(self) -> timedelta:
        if self.sunrise is not None and self.civil_dawn is not None:
            return (
                    datetime.combine(date.today(), self.sunrise, tzinfo=timezone.utc)
                    - datetime.combine(date.today(), self.civil_dawn, tzinfo=timezone.utc)
            )
        return timedelta(0)


# Broker payloads
class EnginePayloadDict(TypedDict):
    """Cf. related BaseModel."""
    engine_uid: str
    address: str


class EnginePayload(BaseModel[EnginePayloadDict]):
    """Minimal info about a Gaia engine needed for Ouranos to register it.

    Used as a payload sent between Gaia and Ouranos.
    """
    engine_uid: str
    address: str


class EcosystemPingDataDict(TypedDict):
    uid: str
    status: bool


class EcosystemPingData(BaseModel[EcosystemPingDataDict]):
    uid: str
    status: bool


class EnginePingPayloadDict(TypedDict):
    engine_uid: str
    timestamp: datetime
    ecosystems: list[EcosystemPingDataDict]


class EnginePingPayload(BaseModel[EnginePingPayloadDict]):
    engine_uid: str
    timestamp: datetime
    ecosystems: list[EcosystemPingData]


class EcosystemPayloadDict(TypedDict):
    """Cf. related BaseModel."""
    uid: str


class EcosystemPayload(BaseModel[EcosystemPayloadDict]):
    """Base payload for sharing data between Gaia and Ouranos.

    Payloads consist of the uid of the ecosystem and data.
    """
    uid: str
    data: Any

    @classmethod
    def from_base(cls, uid: str, base: Any):
        return cls(
            uid=uid,
            data=base,
        )


class PlacesPayloadDict(EcosystemPayloadDict):
    """Payload to send 'Place' from Gaia to Ouranos."""
    data: list[Place]


class PlacesPayload(BaseModel[PlacesPayloadDict], EcosystemPayload):
    """Payload to send 'Place' from Gaia to Ouranos."""
    data: list[Place]


# Config Payload
class BaseInfoConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: BaseInfoConfigDict


class BaseInfoConfigPayload(BaseModel[BaseInfoConfigPayloadDict], EcosystemPayload):
    """Payload to send 'BaseInfoConfig' from Gaia to Ouranos."""
    data: BaseInfoConfig


class ManagementConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: ManagementConfigDict


class ManagementConfigPayload(BaseModel[ManagementConfigPayloadDict], EcosystemPayload):
    """Payload to send 'ManagementConfig' from Gaia to Ouranos."""
    data: ManagementConfig


class ChaosParametersPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: ChaosParametersDict


class ChaosParametersPayload(BaseModel[ChaosParametersPayloadDict], EcosystemPayload):
    """Payload to send 'ChaosParameters' from Gaia to Ouranos."""
    data: ChaosParameters


class NycthemeralCycleConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: NycthemeralCycleConfigDict


class NycthemeralCycleConfigPayload(BaseModel[NycthemeralCycleConfigPayloadDict], EcosystemPayload):
    """Payload to send 'NycthemeralCycleConfig' from Gaia to Ouranos."""
    data: NycthemeralCycleConfig


class ClimateConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: list[ClimateConfigDict]


class ClimateConfigPayload(BaseModel[ClimateConfigPayloadDict], EcosystemPayload):
    """Payload to send a list of 'ClimateConfig' from Gaia to Ouranos."""
    data: list[ClimateConfig] = Field(default_factory=list)


class WeatherConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: list[WeatherConfigDict]


class WeatherConfigPayload(BaseModel[WeatherConfigPayloadDict], EcosystemPayload):
    """Payload to send a list of 'WeatherConfig' from Gaia to Ouranos."""
    data: list[WeatherConfig] = Field(default_factory=list)


class HardwareConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: list[HardwareConfigDict]


class HardwareConfigPayload(BaseModel[HardwareConfigPayloadDict], EcosystemPayload):
    """Payload to send a list of 'HardwareConfig' from Gaia to Ouranos."""
    data: list[HardwareConfig] = Field(default_factory=list)


class PlantConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: list[PlantConfigDict]


class PlantConfigPayload(BaseModel[PlantConfigPayloadDict], EcosystemPayload):
    """Payload to send a list of 'PlantConfig' from Gaia to Ouranos."""
    data: list[PlantConfig] = Field(default_factory=list)


# Data payloads
class SensorsDataPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: SensorsDataDict


class SensorsDataPayload(BaseModel[SensorsDataPayloadDict], EcosystemPayload):
    """Payload to send 'SensorsData' from Gaia to Ouranos."""
    data: SensorsData


class NycthemeralCycleInfoPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: NycthemeralCycleInfoDict


class NycthemeralCycleInfoPayload(BaseModel[NycthemeralCycleInfoPayloadDict], EcosystemPayload):
    """Payload to send 'LightData' from Gaia to Ouranos."""
    data: NycthemeralCycleInfo


class LightDataPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: LightDataDict


class LightDataPayload(BaseModel[LightDataPayloadDict], EcosystemPayload):
    """Payload to send 'LightData' from Gaia to Ouranos."""
    data: LightData


class ActuatorsDataPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: list[ActuatorStateRecord]


class ActuatorsDataPayload(BaseModel[ActuatorsDataPayloadDict], EcosystemPayload):
    """Payload to send 'ActuatorsState' from Gaia to Ouranos."""
    data: list[ActuatorStateRecord] = Field(default_factory=list)


class HealthDataPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: HealthDataDict


class HealthDataPayload(BaseModel[HealthDataPayloadDict], EcosystemPayload):
    """Payload to send 'HealthRecord' from Gaia to Ouranos."""
    data: HealthData


# Actuators payload
class TurnActuatorPayloadDict(TypedDict):
    """Cf. related BaseModel."""
    ecosystem_uid: str
    actuator: HardwareType
    group: NotRequired[str]
    mode: ActuatorModePayload
    level: NotRequired[float]
    countdown: NotRequired[float]


class TurnActuatorPayload(BaseModel[TurnActuatorPayloadDict]):
    """Payload from Ouranos to Gaia to request a change in mode for a type of
    actuator in the given ecosystem.

    :arg ecosystem_uid: the uid of the ecosystem in which a change is requested.
    :arg actuator: the type of actuator to change.
    :arg group: the actuator group to change.
    :arg mode: the desired mode. Rem: it is a 'ActuatorModePayload', not an
              'ActuatorMode'.
    :arg level: the level to which the actuator should be set. Is only used if
                the mode is 'on'.
    :arg countdown: the delay before which to change mode.
    """
    ecosystem_uid: str
    actuator: HardwareType
    group: str = Field(default_factory=lambda data: data["actuator"].name)
    mode: ActuatorModePayload = ActuatorModePayload.automatic
    level: float = Field(default=100.0, ge=0.0, le=100.0)
    countdown: float = Field(default=0.0, ge=0.0)

    @field_validator("actuator", mode="before")
    @classmethod
    def parse_actuator(cls, value):
        if isinstance(value, int):
            return HardwareType(value)
        return safe_enum_from_name(HardwareType, value)

    @field_validator("mode", mode="before")
    @classmethod
    def parse_mode(cls, value):
        return safe_enum_from_name(ActuatorModePayload, value)


class RouteDict(TypedDict):
    """Cf. related BaseModel."""
    engine_uid: str
    ecosystem_uid: str | None


class Route(BaseModel[RouteDict]):
    """Information about on which engine an ecosystem is found."""
    engine_uid: str
    ecosystem_uid: str | None = None


# Crud payload
class CrudPayloadDict(TypedDict):
    """Cf. related BaseModel."""
    uuid: UUID
    routing: RouteDict
    action: CrudAction
    target: str
    kwargs: dict[str, Any]


class CrudPayload(BaseModel[CrudPayloadDict]):
    """Payload from Ouranos to Gaia to request a change in an ecosystem config.

    :arg uuid: an uuid for the CRUD request.
    :arg routing: the type of actuator to change.
    :arg action: the CRUD action that is requested.
    :arg target: the ecosystem configuration target.
    :arg kwargs: the kwargs to be passed to the ecosystem configuration target.
    """
    uuid: UUID = Field(default_factory=uuid4)
    routing: Route
    action: CrudAction
    target: str
    kwargs: dict = Field(default_factory=dict)

    @field_validator("action", mode="before")
    @classmethod
    def parse_action(cls, value):
        return safe_enum_from_name(CrudAction, value)


# Buffered data payloads
class BufferedDataPayloadDict(TypedDict):
    """Cf. related BaseModel."""
    uuid: UUID


class BufferedSensorRecord(NamedTuple):
    """A version of SensorRecord saved by gaia when it could not be sent.

    :arg ecosystem_uid: the uid of the ecosystem in which the measurement was
                        taken.
    :arg sensor_uid: the uid of the sensor that took the measurement.
    :arg measure: the name of the measure taken.
    :arg value: the value of the sensor's measurement.
    :arg timestamp: the timestamp of when the measurement was done.

    Used by Gaia events and as part of a payload sent between Gaia and Ouranos.
    """
    ecosystem_uid: str
    sensor_uid: str
    measure: str
    value: float
    timestamp: datetime


class BufferedActuatorRecord(NamedTuple):
    """A version of ActuatorRecord saved by gaia when it could not be sent.

    :arg ecosystem_uid: the uid of the ecosystem in which the measurement was
                        taken.
    :arg type: the type of the actuator recorded. Not important any more.
    :arg group: the name of the actuator group.
    :arg active: whether the actuator status and mode can be changed or not.
    :arg mode: the actuator mode, either 'automatic' or 'manual'
    :arg status: the actuator status, either 'true' (on) or 'false' (off).
    :arg level: the actuator level, as a percentage of its maximum intensity.
    :arg timestamp: the timestamp of when the measurement was done.

    Used by Gaia events and as part of a payload sent between Gaia and Ouranos.
    """
    ecosystem_uid: str
    type: HardwareType
    group: str
    active: bool
    mode: ActuatorMode
    status: bool
    level: float | None
    timestamp: datetime | None


class BufferedDataPayload(BaseModel[BufferedDataPayloadDict]):
    """Payload to send a list of buffered data from Gaia to Ouranos.

    :arg data: a list of data (DB rows) that could not be sent before.
    :arg uuid: the id of the transaction.
    """
    data: list[BufferedSensorRecord] | list[BufferedActuatorRecord]
    uuid: UUID


class BufferedSensorsDataPayloadDict(BufferedDataPayloadDict):
    """Cf. related BaseModel."""
    data: list[BufferedSensorRecord]


class BufferedSensorsDataPayload(BaseModel[BufferedSensorsDataPayloadDict], BufferedDataPayload):
    """Payload to send a list of 'BufferedSensorRecord' from Gaia to Ouranos.

    :arg data: a list of 'BufferedSensorRecord' that could not be sent before.
    :arg uuid: the id of the transaction.
    """
    data: list[BufferedSensorRecord]
    uuid: UUID


BufferedHealthRecordPayload = BufferedSensorsDataPayload


BufferedHealthRecordPayloadDict = BufferedSensorsDataPayloadDict


class BufferedActuatorsStatePayloadDict(BufferedDataPayloadDict):
    """Cf. related BaseModel."""
    data: list[BufferedActuatorRecord]


class BufferedActuatorsStatePayload(BaseModel[BufferedActuatorsStatePayloadDict], BufferedDataPayload):
    """Payload to send a list of 'BufferedActuatorRecord' from Gaia to Ouranos.

    :arg data: a list of 'BufferedActuatorRecord' that could not be sent before.
    :arg uuid: the id of the transaction.
    """
    data: list[BufferedActuatorRecord]
    uuid: UUID


# Request (CRUD & buffered data saving) results
class Result(StrEnum):
    """The status of a request."""
    success = "success"
    failure = "failure"


class RequestResultDict(TypedDict):
    """Cf. related BaseModel."""
    uuid: str
    status: Result
    message: str | None


class RequestResult(BaseModel[RequestResultDict]):
    """The result of a CRUD request or after sending buffered sensors data.

    :arg uuid: the id of the request treated.
    :arg status: the status of the request.
    :arg message: an optional message containing information (about failure for
                  example).

    Used as an acknowledgment after requests between Gaia and Ouranos.
    """
    uuid: UUID
    status: Result
    message: str | None = None


@overload
def to_identified(config: AnonymousHardwareConfigDict, identifier: dict[str, str]) -> HardwareConfigDict: ...

@overload
def to_identified(config: AnonymousClimateConfigDict, identifier: dict[str, str]) -> ClimateConfigDict: ...

@overload
def to_identified(config: AnonymousPlantConfigDict, identifier: dict[str, str]) -> PlantConfigDict: ...

@overload
def to_identified(config: AnonymousWeatherConfigDict, identifier: dict[str, str]) -> WeatherConfigDict: ...

def to_identified(config: dict, identifier: dict[str, str]) -> dict:
    return {**identifier, **config}


@overload
def to_anonymous(config: HardwareConfigDict, identifier: str) -> AnonymousHardwareConfigDict: ...

@overload
def to_anonymous(config: ClimateConfigDict, identifier: str) -> AnonymousClimateConfigDict: ...

@overload
def to_anonymous(config: PlantConfigDict, identifier: str) -> AnonymousPlantConfigDict: ...

@overload
def to_anonymous(config: WeatherConfigDict, identifier: str) -> AnonymousWeatherConfigDict: ...

def to_anonymous(config: dict, identifier: str) -> dict:
    return {k: v for k, v in config.items() if k != identifier}
