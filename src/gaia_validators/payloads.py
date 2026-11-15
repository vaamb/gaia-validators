from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, NamedTuple, NotRequired, TypedDict
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from gaia_validators.models import (
    ActuatorMode, ActuatorModePayload, ActuatorStateRecord, BaseInfoConfig,
    BaseInfoConfigDict, BaseModel, ChaosParameters, ChaosParametersDict, ClimateConfig,
    ClimateConfigDict, CrudAction, HardwareConfig, HardwareConfigDict, HardwareType,
    HealthData, HealthDataDict, LightData, LightDataDict, ManagementConfig,
    ManagementConfigDict, NycthemeralCycleConfig, NycthemeralCycleConfigDict,
    NycthemeralCycleInfo, NycthemeralCycleInfoDict, Place, PlaceDict, PlantConfig,
    PlantConfigDict, safe_enum_from_name, SensorsData, SensorsDataDict, WeatherConfig,
    WeatherConfigDict,
)


# Broker payloads
class EnginePayloadDict(TypedDict):
    """Cf. related BaseModel."""
    engine_uid: str
    address: str
    contract_version: int


class EnginePayload(BaseModel[EnginePayloadDict]):
    """Minimal info about a Gaia engine needed for Ouranos to register it.

    Used as a payload sent between Gaia and Ouranos.
    """
    engine_uid: str
    address: str
    contract_version: int = 0


class EngineRegistrationAckDict(TypedDict):
    """Cf. related BaseModel."""
    host_uid: str | UUID
    contract_version: int
    status: Result


class EngineRegistrationAck(BaseModel[EngineRegistrationAckDict]):
    """Contract version

    Used as a payload sent between Gaia and Ouranos.
    """
    host_uid: str | UUID
    contract_version: int = 0
    status: Result


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
    data: list[PlaceDict]


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
