from __future__ import annotations

from datetime import datetime, time, timezone
from enum import auto, Enum, IntEnum, IntFlag
from typing import Any, Literal, NamedTuple, Self, Type, TypedDict, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel as _BaseModel, ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass

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


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class LaxBaseModel(_BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


class BaseModel(_BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        populate_by_name=True,
    )


@dataclass()
class Empty:
    """A class for empty data/payload.

    Used by Gaia.
    """
    pass


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
class BaseInfoConfig(BaseModel):
    """Minimum info about a Gaia ecosystem needed for Ouranos to register it.

    Used by Gaia and Ouranos.
    """
    engine_uid: str
    uid: str
    name: str
    status: bool = False


class BaseInfoConfigDict(TypedDict):
    """Cf. related BaseModel."""
    engine_uid: str
    uid: str
    name: str
    status: bool


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
    # available basal management = 16
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


ManagementNames = Literal[*_get_enum_names(ManagementFlags)]  # noqa: works when imported


class ManagementConfig(BaseModel):
    """Long form of a Gaia ecosystem subroutine management info.

    Used by Gaia in the ecosystems configuration file.
    """
    sensors: bool = False
    light: bool = False
    camera: bool = False
    database: bool = False

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
        return cls(**payload)


class ManagementConfigDict(TypedDict):
    """Cf. related BaseModel."""
    sensors: bool
    light: bool
    camera: bool
    database: bool

    alarms: bool
    climate: bool
    watering: bool
    health: bool
    pictures: bool


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


ActuatorModeNames = Literal[*_get_enum_names(ActuatorMode)]  # noqa: works when imported


class ActuatorStatus(StrEnum):
    """Actuator status.

    Used by Gaia and Ouranos.
    """
    on = "on"
    off = "off"


ActuatorStatusNames = Literal[*_get_enum_names(ActuatorStatus)]  # noqa: works when imported


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


# Light
class LightMethod(StrEnum):
    """Lighting method.

    If 'fixed', lights will be on between 'DayConfig.day' and 'DayConfig.night'.
    If 'elongate', lights will be on between 'DayConfig.day' and a bit after
    sunrise, and between a bit before sunset and 'DayConfig.night'. Sunrise and
    sunset times are downloaded from the internet.
    If 'mimic', lights will be on between the sunrise and sunset of a specified
    place.

    Used by Gaia and Ouranos.
    """
    fixed = "fixed"
    elongate = "elongate"
    mimic = "mimic"


LightMethodNames = Literal[*_get_enum_names(LightMethod)]  # noqa: works when imported


# Climate
class ChaosConfig(BaseModel):
    """Chaos parameters.

    :arg frequency: the average delay between two chaotic events. If set to 0,
                     no chaotic event will occur.
    :arg duration: the duration in day of one chaotic event.
    :arg Intensity: the intensity of a chaotic event. It influences the
                     temperature, humidity, light and watering levels.

    Used by Gaia ecosystems configuration file.
    """
    frequency: int = 0
    duration: int = 0
    intensity: float = 0.0


class ChaosConfigDict(TypedDict):
    """Cf. related BaseModel."""
    frequency: int
    duration: int
    intensity: float


class TimeWindow(BaseModel):
    beginning: datetime | None = None
    end: datetime | None = None

    @field_validator("beginning", "end", mode="before")
    def parse_time(cls, value):
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
            dt.astimezone(timezone.utc)
            return dt
        return value


class TimeWindowDict(TypedDict):
    beginning: datetime | None
    end: datetime | None


class ChaosParameters(ChaosConfig):
    time_window: TimeWindow = Field(default_factory=TimeWindow)


class ChaosParametersDict(ChaosConfigDict):
    time_window: TimeWindow | TimeWindowDict


class DayConfig(BaseModel):
    """Info about the day and night times.

    Used by Gaia ecosystems configuration file.

    Rem: if the environment light method used is `LightMethod.elongate` or
    `LightMethod.mimic`, the times given will be overriden.
    """
    day: time | None = time(8)
    night: time | None = time(20)

    @field_validator("day", "night", mode="before")
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
                minutes = 0
            return time(int(hours), int(minutes))


class DayConfigDict(TypedDict):
    """Cf. related BaseModel."""
    day: time
    night: time


class SkyConfig(DayConfig):
    """An augmented version of `DayConfig` with the light method added.

    Used by Gaia ecosystems configuration file.
    """
    lighting: LightMethod = LightMethod.fixed
    target: str | None = None

    @field_validator("lighting", mode="before")
    def parse_lighting(cls, value):
        return safe_enum_from_name(LightMethod, value)


class SkyConfigDict(TypedDict):
    """Cf. related BaseModel."""
    day:  time | None | str
    night:  time | None | str
    lighting: str
    target: str | None


class ClimateParameter(StrEnum):
    """Climate parameters that can be controlled by Gaia ecosystem subroutines.

    Used by Gaia and Ouranos.
    """
    temperature = "temperature"
    humidity = "humidity"
    light = "light"
    wind = "wind"


ClimateParameterNames = Literal[*_get_enum_names(ClimateParameter)]  # noqa: works when imported


class AnonymousClimateConfig(BaseModel):
    """Configuration for controlling one climatic parameter.

    Used by Gaia in the ecosystems configuration file.

    Rem: AnonymousClimateConfig does not store the parameter name and should
    only be used when its named is linked to the config (for ex: in a dict)
    """
    day: float
    night: float
    hysteresis: float = 0.0
    alarm: float | None = None

    @field_validator("day", "night", "hysteresis", mode="before")
    def cast_as_float(cls, value):
        return float(value)

    @field_validator("alarm", mode="before")
    def cast_as_float_or_none(cls, value):
        if value is None:
            return None
        return float(value)


class AnonymousClimateConfigDict(TypedDict):
    """Cf. related BaseModel."""
    day: float
    night: float
    hysteresis: float
    alarm: float | None


class ClimateConfig(AnonymousClimateConfig):
    """Configuration info for a single climate parameter.

    Used by Ouranos.
    """
    parameter: ClimateParameter

    @field_validator("parameter", mode="before")
    def parse_parameter(cls, value):
        return safe_enum_from_name(ClimateParameter, value)


class ClimateConfigDict(AnonymousClimateConfigDict):
    """Cf. related BaseModel."""
    parameter: ClimateParameter


class EnvironmentConfig(BaseModel):
    """Configuration for the control of the environment of one ecosystem.

    For the detail of the various attributes, see the related models.

    Used as part of a payload sent between Gaia and Ouranos.
    """
    chaos: ChaosConfig = Field(default_factory=ChaosConfig)
    sky: SkyConfig = Field(default_factory=SkyConfig)
    climate: list[ClimateConfig] = Field(default_factory=list)

    @field_validator("climate", mode="before")
    def format_climate(cls, value: dict | list):
        if isinstance(value, dict):
            return [{"parameter": key, **value} for key, value in value.items()]
        return value


class EnvironmentConfigDict(TypedDict):
    """Cf. related BaseModel."""
    chaos: ChaosConfigDict
    sky: SkyConfigDict
    climate: list[ClimateConfigDict]


# Hardware
class HardwareLevel(StrEnum):
    """Level at which the hardware operates."""

    environment = "environment"
    plants = "plants"


HardwareLevelNames = Literal[*_get_enum_names(HardwareLevel)]  # noqa: works when imported


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


HardwareTypeNames = Literal[*_get_enum_names(HardwareType)]  # noqa: works when imported


class Measure(BaseModel):
    name: str
    unit: str | None = None


class MeasureDict(TypedDict):
    name: str
    unit: str | None


class AnonymousHardwareConfig(BaseModel):
    """Configuration info for a single unidentified piece of hardware.

    Used by Gaia in the ecosystems configuration file.

    Rem: HardwareConfig does not store the hardware uid and should only be used
    when its id is linked to the config (for ex: in a dict)
    """
    name: str
    address: str
    type: HardwareType
    level: HardwareLevel
    model: str
    measures: list[Measure] = Field(default_factory=list, validation_alias="measure")
    plants: list[str] = Field(default_factory=list, validation_alias="plant")
    multiplexer_model: str | None = Field(default=None, validation_alias="multiplexer")

    @field_validator("type", mode="before")
    def parse_type(cls, value):
        if isinstance(value, int):
            return HardwareType(value)
        return safe_enum_from_name(HardwareType, value)

    @field_validator("level", mode="before")
    def parse_level(cls, value):
        return safe_enum_from_name(HardwareLevel, value)

    @field_validator("measures", mode="before")
    def parse_measures(cls, value: str | list[str] | None):
        if value is None:
            return []
        if isinstance(value, str):
            value = value
        rv = []
        for v in value:
            if isinstance(v, str):
                v_split = v.split("|")
                name = v_split[0]
                try:
                    unit = v_split[1]
                except IndexError:
                    unit = None
                v = {"name": name, "unit": unit}
            rv.append(v)
        return rv

    @field_validator("plants", mode="before")
    def parse_plants(cls, value: str | list[str] | None):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class AnonymousHardwareConfigDict(TypedDict):
    """Cf. related BaseModel."""
    name: str
    address: str
    type: HardwareType
    level: HardwareLevel
    model: str
    measures: list[MeasureDict]
    plants: list[str]
    multiplexer_model: str | None


class HardwareConfig(AnonymousHardwareConfig):
    """Configuration info for a single piece of hardware.

    Used by Ouranos.
    """
    uid: str


class HardwareConfigDict(AnonymousHardwareConfigDict):
    """Cf. related BaseModel."""
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


class SensorsData(BaseModel):
    """A collection of all the sensor measurements for one ecosystem at one time
    point.

    For the detail of the various attributes, see the related models.

    Used by Gaia sensors subroutine and as part of a payload sent between Gaia
    and Ouranos."""
    timestamp: datetime = Field(default_factory=_now)
    records: list[SensorRecord] = Field(default_factory=list)
    average: list[MeasureAverage] = Field(default_factory=list)
    alarms: list[SensorAlarm] = Field(default_factory=list)


class SensorsDataDict(TypedDict):
    """Cf. related BaseModel."""
    timestamp: datetime | str
    records: list[SensorRecord]
    average: list[MeasureAverage]
    alarms: list[SensorAlarm]


class HealthRecord(NamedTuple):
    """The ecosystem vegetation health measurement for one ecosystem at one time
    point.

    :arg green: the percentage of green pixels.
    :arg necrosis: the percentage of pixels corresponding to plants showing
                    signs of chlorosis and/or necrosis.
    :arg index: the health index.
    :arg timestamp: the timestamp of when the measurement was done.

    Used by Gaia health subroutine and as part of a payload sent between Gaia
    and Ouranos.
    """
    green: float
    necrosis: float
    index: float
    timestamp: datetime


class LightingHours(BaseModel):
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


class LightingHoursDict(TypedDict):
    """Cf. related BaseModel."""
    morning_start: time | str
    morning_end: time | None | str
    evening_start: time | None | str
    evening_end: time | str


class LightData(LightingHours):
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


class LightDataDict(LightingHoursDict):
    """Cf. related BaseModel."""
    method: LightMethod


# Actuators data
class ActuatorState(BaseModel):
    """The state of one (type of) actuator.

    Used by Gaia and Ouranos api.
    """
    active: bool = False
    status: bool = False
    level: float | None = None
    mode: ActuatorMode = ActuatorMode.automatic

    @field_validator("mode", mode="before")
    def parse_mode(cls, value):
        return safe_enum_from_name(ActuatorMode, value)


class ActuatorStateDict(TypedDict):
    """Cf. related BaseModel."""
    active: bool
    status: bool
    level: float | None
    mode: ActuatorMode


class ActuatorsData(BaseModel):
    """A compilation of all the types of actuator and their state

    Used by Gaia and as part of a payload sent between Gaia and Ouranos.
    """
    light: ActuatorState = ActuatorState()
    cooler: ActuatorState = ActuatorState()
    heater: ActuatorState = ActuatorState()
    humidifier: ActuatorState = ActuatorState()
    dehumidifier: ActuatorState = ActuatorState()
    fan: ActuatorState = ActuatorState()


class ActuatorsDataDict(TypedDict):
    """Cf. related BaseModel."""
    light: ActuatorStateDict
    cooler: ActuatorStateDict
    heater: ActuatorStateDict
    humidifier: ActuatorStateDict
    dehumidifier: ActuatorStateDict
    fan: ActuatorsDataDict


# Places
class Coordinates(NamedTuple):
    latitude: float
    longitude: float


class Place(BaseModel):
    name: str
    coordinates: Coordinates


class PlaceDict(TypedDict):
    name: str
    coordinates: Coordinates


# Sun times
class SunTimes(LaxBaseModel):
    """Information about sunrise and sunset events for a given place.

    Used by Gaia.
    """
    civil_dawn: time = Field(validation_alias="civil_twilight_begin")
    sunrise: time
    sunset: time
    civil_dusk: time = Field(validation_alias="civil_twilight_end")

    @field_validator("civil_dawn", "sunrise", "sunset", "civil_dusk", mode="before")
    def parse_time(cls, value):
        if isinstance(value, str):
            try:
                dt = datetime.strptime(value, "%H:%M:%S")
            except ValueError:
                dt = datetime.strptime(value, "%I:%M:%S %p")
            return dt.time()
        return value


class SunTimesDict(TypedDict):
    """Cf. related BaseModel."""
    civil_dawn: time
    sunrise: time
    sunset: time
    civil_dusk: time


# Broker payloads
class EnginePayload(BaseModel):
    """Minimal info about a Gaia engine needed for Ouranos to register it.

    Used as a payload sent between Gaia and Ouranos.
    """
    engine_uid: str
    address: str


class EnginePayloadDict(TypedDict):
    """Cf. related BaseModel."""
    engine_uid: str
    address: str


class EcosystemPayload(BaseModel):
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


class EcosystemPayloadDict(TypedDict):
    """Cf. related BaseModel."""
    uid: str


class PlacesPayload(EcosystemPayload):
    """Payload to send 'Place' from Gaia to Ouranos."""
    data: list[Place]


class PlacesPayloadDict(EcosystemPayloadDict):
    """Payload to send 'Place' from Gaia to Ouranos."""
    data: list[Place]


# Config Payload
class BaseInfoConfigPayload(EcosystemPayload):
    """Payload to send 'BaseInfoConfig' from Gaia to Ouranos."""
    data: BaseInfoConfig


class BaseInfoConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: BaseInfoConfigDict


class ManagementConfigPayload(EcosystemPayload):
    """Payload to send 'ManagementConfig' from Gaia to Ouranos."""
    data: ManagementConfig


class ManagementConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: ManagementConfigDict


class EnvironmentConfigPayload(EcosystemPayload):
    """Payload to send 'EnvironmentConfig' from Gaia to Ouranos."""
    data: EnvironmentConfig


class EnvironmentConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: EnvironmentConfigDict


class HardwareConfigPayload(EcosystemPayload):
    """Payload to send a list of 'IdentifiedHardwareConfig' from Gaia to
    Ouranos."""
    data: list[HardwareConfig]


class HardwareConfigPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: list[HardwareConfigDict]


# Data payloads
class SensorsDataPayload(EcosystemPayload):
    """Payload to send 'SensorsData' from Gaia to Ouranos."""
    data: SensorsData


class SensorsDataPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: SensorsDataDict


class LightDataPayload(EcosystemPayload):
    """Payload to send 'LightData' from Gaia to Ouranos."""
    data: LightData


class LightDataPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: LightDataDict


class ActuatorsDataPayload(EcosystemPayload):
    """Payload to send 'ActuatorsData' from Gaia to Ouranos."""
    data: ActuatorsData


class ActuatorsDataPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: ActuatorsDataDict


class HealthDataPayload(EcosystemPayload):
    """Payload to send 'HealthRecord' from Gaia to Ouranos."""
    data: HealthRecord


class HealthDataPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: HealthRecord


class ChaosParametersPayload(EcosystemPayload):
    """Payload to send 'ChaosParameters' from Gaia to Ouranos."""
    data: ChaosParameters


class ChaosParametersPayloadDict(EcosystemPayloadDict):
    """Cf. related BaseModel."""
    data: ChaosParametersDict


# Actuators payload
class TurnActuatorPayload(BaseModel):
    """Payload from Ouranos to Gaia to request a change in mode for a type of
    actuator in the given ecosystem.
    
    :arg ecosystem_uid: the uid of the ecosystem in which a change is requested.
    :arg actuator: the type of actuator to change.
    :arg mode: the desired mode. Rem: it is a 'ActuatorModePayload', not a
              'ActuatorMode'.
    :arg countdown: the delay before which to change mode.
    """
    ecosystem_uid: str | None = None  # can be None if transferred in parallel
    actuator: HardwareType
    mode: ActuatorModePayload = ActuatorModePayload.automatic
    countdown: float = 0.0

    @field_validator("actuator", mode="before")
    def parse_actuator(cls, value):
        if isinstance(value, int):
            return HardwareType(value)
        return safe_enum_from_name(HardwareType, value)

    @field_validator("mode", mode="before")
    def parse_mode(cls, value):
        return safe_enum_from_name(ActuatorModePayload, value)


class TurnActuatorPayloadDict(TypedDict):
    """Cf. related BaseModel."""
    ecosystem_uid: str | None
    actuator: HardwareType
    mode: ActuatorModePayload
    countdown: float


class Route(BaseModel):
    """Information about on which engine an ecosystem is found."""
    engine_uid: str
    ecosystem_uid: str | None = None


class RouteDict(TypedDict):
    """Cf. related BaseModel."""
    engine_uid: str
    ecosystem_uid: str | None


# Crud payload
class CrudPayload(BaseModel):
    """Payload from Ouranos to Gaia to request a change in an ecosystem config.

    :arg uuid: an uuid for the CRUD request.
    :arg routing: the type of actuator to change.
    :arg action: the CRUD action that is requested.
    :arg target: the ecosystem configuration target.
    :arg data: the data to be passed to the ecosystem configuration target.
    """
    uuid: UUID = Field(default_factory=uuid4)
    routing: Route
    action: CrudAction
    target: str
    data: str | dict = Field(default_factory=dict)

    @field_validator("action", mode="before")
    def parse_action(cls, value):
        return safe_enum_from_name(CrudAction, value)


class CrudPayloadDict(TypedDict):
    """Cf. related BaseModel."""
    uuid: str
    routing: RouteDict
    action: CrudAction
    target: str
    data: str | dict


# Buffered data payloads
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


class BufferedSensorsDataPayload(BaseModel):
    """Payload to send a list of 'BufferedSensorRecord' from Gaia to Ouranos.

    :arg data: a list of 'BufferedSensorRecord' that could not be sent before.
    :arg uuid: the id of the transaction.
    """
    data: list[BufferedSensorRecord]
    uuid: UUID


class BufferedSensorsDataPayloadDict(TypedDict):
    """Cf. related BaseModel."""
    data: list[BufferedSensorRecord]
    uuid: UUID


# Request (CRUD & buffered data saving) results
class Result(StrEnum):
    """The status of a request."""
    success = "success"
    failure = "failure"


class RequestResult(BaseModel):
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


class RequestResultDict(TypedDict):
    """Cf. related BaseModel."""
    uuid: str
    status: Result
    message: str | None


_imported = {
    "_BaseModel", "annotations", "Any", "dataclass", "datetime", "EnumType",
    "Field", "field_validator", "IntFlag", "Literal", "StrEnum", "time",
    "TypedDict", "UUID", "uuid4"
}

__all__ = [_ for _ in dir() if _ not in ["_imported", *_imported, *__builtins__]]
