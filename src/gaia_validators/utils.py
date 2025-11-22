from datetime import date, datetime, time, timedelta, timezone
from math import acos, asin, cos, pi, sin

from pydantic.json_schema import models_json_schema

import gaia_validators as gv


def get_sun_times(
        coordinates: gv.Coordinates,
        day: date | None = None,
) -> gv.SunTimes:
    latitude = coordinates.latitude
    longitude = coordinates.longitude
    TO_RAD = pi / 180.0
    # from https://en.wikipedia.org/wiki/Sunrise_equation
    # Day of year
    today: date = day or date.today()
    day_of_year: float = float(today.timetuple().tm_yday)
    # Correct for leap seconds and the fact that julian day start at noon on jan 1 (= 0.5 day)
    day_of_year += 0.0008 - 0.5
    # Mean solar time
    solar_noon: float = day_of_year - (longitude / 360.0)
    # Solar mean Anomaly
    sma_deg: float = (357.5291 + (0.98560028 * solar_noon)) % 360.0
    # Equation of center
    center_deg: float = (
        1.9148 * sin(TO_RAD * sma_deg)
        + 0.02 * sin(2.0 * TO_RAD * sma_deg)
        + 0.0003 * sin(3.0 * TO_RAD * sma_deg)
    )
    # Ecliptic longitude
    el_deg: float = (sma_deg + center_deg + 180.0 + 102.9372) % 360.0
    # Solar transit
    st = (
        solar_noon
        + (0.0053 * sin(TO_RAD * sma_deg))
        - (0.0069 * sin(2 * TO_RAD * el_deg))
    )
    # Declination of the sun
    sin_decl: float = sin(TO_RAD * el_deg) * sin(TO_RAD * 23.4397)
    cos_decl: float = cos(asin(sin_decl))

    # Hour angle
    def get_hour_angle(angle: float) -> float | None:
        cos_ha = (
            (sin(TO_RAD * angle) - (sin(TO_RAD * latitude * sin_decl)))
            / (cos(TO_RAD * latitude) * cos_decl)
        )
        if cos_ha < -1 or cos_ha > 1:
            return None
        return 1 / TO_RAD * (acos(cos_ha))

    # Some specific sun angles
    sun_angles = {
        -0.833: ("sunrise", "sunset"),
        -6.0: ("civil_dawn", "civil_dusk"),
        -12.0: ("nautical_dawn", "nautical_dusk"),
        -18.0: ("astronomical_dawn", "astronomical_dusk"),
    }

    hour_angles = {
        meaning: get_hour_angle(angle) for angle, meaning in sun_angles.items()
    }

    times: dict[str, float | None] = {
        "solar_noon": st,
    }
    for meaning, ha in hour_angles.items():
        if ha is None:
            times[meaning[0]] = None
            times[meaning[1]] = None
        else:
            times[meaning[0]] = st - ha / 360
            times[meaning[1]] = st + ha / 360

    def day_of_year_to_time(year: int, doy: float | None) -> time | None:
        if doy is None:
            return None
        secs_per_day = 60 * 60 * 24
        days = doy // 1
        secs = (doy % 1) * secs_per_day
        raw_dt = datetime(year, 1, 1) + timedelta(days=days, seconds=secs)
        utc_dt = raw_dt.replace(tzinfo=timezone.utc)
        dt = utc_dt.astimezone()
        return dt.time()

    return gv.SunTimes.model_validate(
        {
            "datestamp": day,
            **{
                meaning: day_of_year_to_time(today.year, doy)
                for meaning, doy in times.items()
            }
        }
    )


def generate_json_schemas() -> dict:
    _, schemas = models_json_schema([
        # Engine payload
        (gv.EnginePingPayload, "validation"),
        (gv.PlacesPayload, "validation"),
        # Ecosystem config payloads
        (gv.BaseInfoConfigPayload, "validation"),
        (gv.ManagementConfigPayload, "validation"),
        (gv.ChaosParametersPayload, "validation"),
        (gv.NycthemeralCycleConfigPayload, "validation"),
        (gv.ClimateConfigPayload, "validation"),
        (gv.WeatherConfigPayload, "validation"),
        (gv.HardwareConfigPayload, "validation"),
        (gv.PlantConfigPayload, "validation"),
        # Data payloads
        (gv.SensorsDataPayload, "validation"),
        (gv.NycthemeralCycleInfoPayload, "validation"),
        (gv.LightDataPayload, "validation"),
        (gv.ActuatorsDataPayload, "validation"),
        (gv.HealthDataPayload, "validation"),
        # Interaction payloads
        (gv.TurnActuatorPayload, "validation"),
        (gv.CrudPayload, "validation"),
        (gv.RequestResult, "validation"),
        # Buffered data payload
        (gv.BufferedDataPayload, "validation"),
        # BufferedHealthRecordPayload = BufferedSensorsDataPayload
        (gv.BufferedActuatorsStatePayload, "validation"),
    ])

    return schemas
