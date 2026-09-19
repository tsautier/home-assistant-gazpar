import logging
from typing import Any, Union

from homeassistant.components.sensor.const import (
    ATTR_STATE_CLASS,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.const import (
    ATTR_ATTRIBUTION,
    ATTR_DEVICE_CLASS,
    ATTR_FRIENDLY_NAME,
    ATTR_ICON,
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_USERNAME,
    UnitOfEnergy,
)
from pygazpar.enum import Frequency, PropertyName  # type: ignore

_LOGGER = logging.getLogger(__name__)

HA_ATTRIBUTION = "Data provided by GrDF"

ICON_GAS = "mdi:fire"

SENSOR_FRIENDLY_NAME = "Gazpar"

LAST_INDEX = -1

ATTR_PCE = "pce"
ATTR_VERSION = "version"
ATTR_ERROR_MESSAGES = "errorMessages"


# --------------------------------------------------------------------------------------------
class Util:

    # Tolerance (kWh) between the index-implied energy of the most recent daily reading
    # (end_index_m3 - start_index_m3) * converter_factor and its independently-reported
    # energy_kwh. GRDF's own metering noise stays under ~10 kWh/day (observed max 9.75 kWh
    # over a 26-day real export); the 2024-10-14 incident's corrupted reading produced a gap
    # of ~52837 kWh. 50 kWh keeps a wide margin on both sides without needing to scale with
    # installation size, since it compares two figures for the *same* day rather than
    # capping absolute daily volume.
    LAST_READING_ENERGY_EPSILON_KWH = 50.0

    # ----------------------------------
    @staticmethod
    def toState(pygazparData: dict[str, list[dict[str, Any]]]) -> Union[float, None]:
        """Compute the cumulative energy state from the daily readings.

        Walks backward from the most recent day while start_index_m3 == end_index_m3
        (no index movement yet), accumulating energy_kwh for those "flat" days -- GRDF
        also reports a small independently-metered energy_kwh on low-consumption days
        even when the 1 m3-resolution index hasn't moved, so this is not dropped. It
        then uses the index of the day where it stops (index * converter_factor) as the
        base, clamping to the last day of the window if every day was flat.

        Before running that walk, the single most recent daily reading is checked for
        internal consistency: GRDF may not have finalized it yet, and a corrupted
        end_index_m3 on just this one record was the root cause of the 2024-10-14
        incident (a single bad reading inflated the state by ~52840 kWh, although the
        official GRDF export shows a normal, continuous index for that day). If the
        index-implied energy for that record disagrees with its own reported energy_kwh
        by more than LAST_READING_ENERGY_EPSILON_KWH, or if its index is simply missing,
        it is dropped and the walk proceeds from the previous (already-published) day
        instead -- older records are never second-guessed this way.
        """

        res = None

        if len(pygazparData) > 0:

            dailyData = pygazparData[Frequency.DAILY.value]

            if dailyData is not None and len(dailyData) > 0:
                dailyData = Util._dropImplausibleMostRecentReading(dailyData)

            if dailyData is not None and len(dailyData) > 0:
                currentIndex = 0
                cumulativeEnergy = 0.0

                # For low consumption, we also use the energy column in addition to the volume index columns
                # and compute more accurately the consumed energy.
                startIndex = dailyData[currentIndex][PropertyName.START_INDEX.value]
                endIndex = dailyData[currentIndex][PropertyName.END_INDEX.value]

                while (
                    (startIndex is not None)
                    and (endIndex is not None)
                    and (currentIndex < len(dailyData))
                    and (float(startIndex) == float(endIndex))
                ):
                    energy = dailyData[currentIndex][PropertyName.ENERGY.value]
                    if energy is not None:
                        cumulativeEnergy += float(energy)
                    currentIndex += 1
                    if currentIndex < len(dailyData):
                        startIndex = dailyData[currentIndex][PropertyName.START_INDEX.value]
                        endIndex = dailyData[currentIndex][PropertyName.END_INDEX.value]

                currentIndex = min(currentIndex, len(dailyData) - 1)

                endIndex = dailyData[currentIndex][PropertyName.END_INDEX.value]
                converterFactorStr = dailyData[currentIndex][PropertyName.CONVERTER_FACTOR.value]

                if endIndex is not None:
                    volumeEndIndex = float(endIndex)
                else:
                    raise ValueError("End index is missing in the daily data.")

                if converterFactorStr is not None:
                    converterFactor = float(converterFactorStr)
                else:
                    raise ValueError("Converter factor is missing in the daily data.")

                res = volumeEndIndex * converterFactor + cumulativeEnergy

        return res

    # ----------------------------------
    @staticmethod
    def _dropImplausibleMostRecentReading(dailyData: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop the most recent (index 0) daily reading if it looks corrupted or unfinalized.

        Older, already-published readings are never touched here -- they keep going
        through the existing backward-walk logic unchanged, including its handling of
        genuine low-consumption flat days.
        """

        mostRecent = dailyData[0]

        startIndexRaw = mostRecent[PropertyName.START_INDEX.value]
        endIndexRaw = mostRecent[PropertyName.END_INDEX.value]

        if startIndexRaw is None or endIndexRaw is None:
            _LOGGER.debug("Ignoring most recent daily reading, missing index: %s", mostRecent)
            return dailyData[1:]

        converterFactorStr = mostRecent[PropertyName.CONVERTER_FACTOR.value]
        energyRaw = mostRecent[PropertyName.ENERGY.value]

        if converterFactorStr is None or energyRaw is None:
            # Can't cross-check consistency without both figures -- let the existing
            # algorithm handle this record as before (it will raise if it lands on it
            # with a missing converter factor).
            return dailyData

        impliedEnergy = (float(endIndexRaw) - float(startIndexRaw)) * float(converterFactorStr)
        gap = abs(impliedEnergy - float(energyRaw))

        if gap >= Util.LAST_READING_ENERGY_EPSILON_KWH:
            _LOGGER.warning(
                "Ignoring most recent daily reading: index-implied energy %.2f kWh does not "
                "match reported energy_kwh %.2f kWh (gap %.2f kWh >= threshold %.2f kWh), "
                "falling back to the previous day: %s",
                impliedEnergy,
                energyRaw,
                gap,
                Util.LAST_READING_ENERGY_EPSILON_KWH,
                mostRecent,
            )
            return dailyData[1:]

        return dailyData

    # ----------------------------------
    @staticmethod
    def toAttributes(
        username: str,
        pceIdentifier: str,
        version: str,
        pygazparData: dict[str, list[dict[str, Any]]],
        errorMessages: list[str],
    ) -> dict[str, Any]:

        res = {
            ATTR_ATTRIBUTION: HA_ATTRIBUTION,
            ATTR_VERSION: version,
            CONF_USERNAME: username,
            ATTR_PCE: pceIdentifier,
            ATTR_UNIT_OF_MEASUREMENT: UnitOfEnergy.KILO_WATT_HOUR,
            ATTR_FRIENDLY_NAME: SENSOR_FRIENDLY_NAME,
            ATTR_ICON: ICON_GAS,
            ATTR_DEVICE_CLASS: SensorDeviceClass.ENERGY,
            ATTR_STATE_CLASS: SensorStateClass.TOTAL_INCREASING,
            ATTR_ERROR_MESSAGES: errorMessages,
            str(Frequency.HOURLY): list[dict[str, Any]](),
            str(Frequency.DAILY): list[dict[str, Any]](),
            str(Frequency.WEEKLY): list[dict[str, Any]](),
            str(Frequency.MONTHLY): list[dict[str, Any]](),
            str(Frequency.YEARLY): list[dict[str, Any]](),
        }

        if len(pygazparData) > 0:
            for frequency in Frequency:
                data = pygazparData.get(frequency.value)

                if data is not None and len(data) > 0:
                    res[str(frequency)] = data
                else:
                    res[str(frequency)] = []

        return res  # type: ignore
