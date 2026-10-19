# 0003. Humidity from the weather station, kappa by blocked CV, scale without intercept

Status: accepted

## Context

Optical PM sensors such as the SDS011 measure light scattered by particles. In damp air
particles take up water and grow, so the reading depends on humidity. Most Sensor.Community
nodes carry their own humidity sensor (DHT22 or BME280), and the usual correction uses it.

Measured in this repo (`results/metrics.json`, flag `f_climate_offset`): the network's own
humidity sensors are more than 15 points away from the DWD synoptic station in 36% to 50%
of location-hours. DHT22 sensors saturate near 100% in winter and drift with age.

## Decision

1. Use the hourly relative humidity of DWD station 4928 (Stuttgart-Schnarrenberg) for the
   correction. City-scale humidity varies much less than the error of a cheap sensor.
   The network's humidity sensors are still checked and reported, but they do not gate
   the PM value.
2. Correct with the kappa-Koehler growth factor `1 + kappa * aw / (1 - aw)`, aw capped at
   0.95. kappa is chosen by blocked two-fold cross validation inside the calibration
   window (first half vs second half), not by in-sample error. In-sample error picks
   kappa = 0 because the scale absorbs everything; blocked CV picked 0.025.
3. Fit one scale factor through the origin after the correction. A slope plus intercept
   fits the calibration window slightly better but the intercept learned in January
   (about 4.3 ug/m3) is a winter background that does not exist in October: that variant
   is worse than doing nothing on the October window.
4. Fog hours (station RH at or above 95%) are corrected and published, not dropped.
   Dropping them would remove 9% to 23% of hours, and the corrected error in that band is
   not worse than in the others on the held-out windows.

## Consequences

- One parameter (kappa) and one scale, both fitted on two weeks of January, transfer to
  a different winter fortnight and to October without refitting.
- The model needs a weather station within a few kilometres. Without one the pipeline
  still runs QC but should not apply the humidity term.
