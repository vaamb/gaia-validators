from unittest import TestCase

from pydantic_core import ValidationError

import gaia_validators as gv


class TestHardware(TestCase):
    def _get_hardware_cfg(self) -> gv.HardwareConfigDict:
        return {
            "uid": "super_uid",
            "name": "test",
            "active": False,
            "address": "address",
            "type": "actuator",
            "level": "ecosystem",
            "groups": ["riser", "actuator"],
            "model": "super_actuator",
            "measures": ["temperature|°C"],
            "plants": ["baobab", "adenium"],
            "multiplexer_model": "super_multiplexer",
        }

    def test_base(self):
        hardware = gv.HardwareConfig(**self._get_hardware_cfg())

        self.assertEqual(hardware.uid, "super_uid")
        self.assertEqual(hardware.name, "test")
        self.assertEqual(hardware.active, False)
        self.assertEqual(hardware.address, "address")
        self.assertIs(hardware.type, gv.HardwareType.actuator)
        self.assertIs(hardware.level, gv.HardwareLevel.ecosystem)
        self.assertEqual(hardware.groups, ["actuator", "riser"])  # Ordered alphabetically
        self.assertEqual(hardware.model, "super_actuator")
        self.assertEqual(hardware.measures, [gv.Measure(name="temperature", unit="°C")])
        self.assertEqual(hardware.plants, ["adenium", "baobab"])  # Ordered alphabetically
        self.assertEqual(hardware.multiplexer_model, "super_multiplexer")

    def test_nones(self):
        hardware_cfg = self._get_hardware_cfg()

        hardware_cfg["groups"] = None
        hardware_cfg["measures"] = None
        hardware_cfg["plants"] = None
        hardware_cfg["multiplexer_model"] = None

        hardware = gv.HardwareConfig(**hardware_cfg)

        self.assertEqual(hardware.groups, ["actuator"])
        self.assertEqual(hardware.measures, [])
        self.assertEqual(hardware.plants, [])
        self.assertEqual(hardware.multiplexer_model, None)

    def test_missing_values(self):
        hardware_cfg = self._get_hardware_cfg()
        hardware_cfg.pop("active")
        hardware_cfg.pop("groups")
        hardware_cfg.pop("measures")
        hardware_cfg.pop("plants")
        hardware_cfg.pop("multiplexer_model")

        hardware = gv.HardwareConfig(**hardware_cfg)

        self.assertIs(hardware.active, True)
        self.assertEqual(hardware.groups, ["actuator"])
        self.assertEqual(hardware.measures, [])
        self.assertEqual(hardware.plants, [])
        self.assertEqual(hardware.multiplexer_model, None)


class TestClimate(TestCase):
    def _get_climate_cfg(self) -> gv.ClimateConfigDict:
        return {
            "parameter": "temperature",
            "day": 25.0,
            "night": 20.0,
            "hysteresis": 2.0,
            "alarm": 5.0,
            "linked_actuators": {
                "increase": "increase_uid",
                "decrease": "decrease_uid",
            },
            "linked_measure": "°C",
        }

    def test_base(self):
        climate = gv.ClimateConfig(**self._get_climate_cfg())

        self.assertIs(climate.parameter, gv.ClimateParameter.temperature)
        self.assertEqual(climate.day, 25.0)
        self.assertEqual(climate.night, 20.0)
        self.assertEqual(climate.hysteresis, 2.0)
        self.assertEqual(climate.alarm, 5.0)
        self.assertEqual(
            climate.linked_actuators,
            gv.ActuatorCouple(increase="increase_uid", decrease="decrease_uid"))
        self.assertEqual(climate.linked_measure, "°C")

    def test_missing_values(self):
        climate_cfg = self._get_climate_cfg()
        climate_cfg.pop("hysteresis")
        climate_cfg.pop("alarm")
        climate_cfg.pop("linked_actuators")
        climate_cfg.pop("linked_measure")

        climate = gv.ClimateConfig(**climate_cfg)

        self.assertEqual(climate.hysteresis, 0.0)
        self.assertEqual(climate.alarm, None)
        self.assertEqual(climate.linked_actuators, None)
        self.assertEqual(climate.linked_measure, None)


class TestWeather(TestCase):
    def _get_weather_cfg(self) -> gv.WeatherConfigDict:
        return {
            "parameter": "rain",
            "pattern": "* * * * *",
            "duration": 300.0,
            "level": 75.0,
            "linked_actuator": "increase_uid",
        }

    def test_base(self):
        weather = gv.WeatherConfig(**self._get_weather_cfg())

        self.assertIs(weather.parameter, gv.WeatherParameter.rain)
        self.assertEqual(weather.pattern, "* * * * *")
        self.assertEqual(weather.duration, 300.0)
        self.assertEqual(weather.level, 75.0)
        self.assertEqual(weather.linked_actuator, "increase_uid")

    def test_missing_values(self):
        weather_cfg = self._get_weather_cfg()

        weather_cfg.pop("level")
        weather_cfg.pop("linked_actuator")

        weather = gv.WeatherConfig(**weather_cfg)

        self.assertEqual(weather.level, 100.0)
        self.assertEqual(weather.linked_actuator, None)

    def test_regex(self):
        weather_cfg = self._get_weather_cfg()

        # The 5 groups are checked with the same rules
        weather_cfg["pattern"] = "*/8 * * * *"
        gv.WeatherConfig(**weather_cfg)

        weather_cfg["pattern"] = "* 1,2,3,5,8,13,21 * * *"
        gv.WeatherConfig(**weather_cfg)

        weather_cfg["pattern"] = "* * 2-4 * *"
        gv.WeatherConfig(**weather_cfg)

        weather_cfg["pattern"] = "*-8 * * * *"
        with self.assertRaises(ValidationError):
            gv.WeatherConfig(**weather_cfg)


class TestClamps(TestCase):
    def test_chaos(self):
        with self.assertRaises(ValidationError):
            gv.ChaosConfig(
                frequency=-1,
            )

        with self.assertRaises(ValidationError):
            gv.ChaosConfig(
                duration=-1,
            )

        with self.assertRaises(ValidationError):
            gv.ChaosConfig(
                intensity=-1.0,
            )

    def test_turn_actuator(self):
        with self.assertRaises(ValidationError):
            gv.TurnActuatorPayload(
                actuator="actuator",
                level=-1.0,
            )

        with self.assertRaises(ValidationError):
            gv.TurnActuatorPayload(
                actuator="actuator",
                level=101.0,
            )

        with self.assertRaises(ValidationError):
            gv.TurnActuatorPayload(
                actuator="actuator",
                countdown=-1.0,
            )

    def test_weather(self):
        with self.assertRaises(ValidationError):
            gv.WeatherConfig(
                parameter="rain",
                pattern="* * * * *",
                duration=0.0,
            )

        with self.assertRaises(ValidationError):
            gv.WeatherConfig(
                parameter="rain",
                pattern="* * * * *",
                duration=1,
                level=-1.0,
            )

        with self.assertRaises(ValidationError):
            gv.WeatherConfig(
                parameter="rain",
                pattern="* * * * *",
                duration=1,
                level=101.0,
            )
