# AeroCal

**Created by Tabish Syed**

AeroCal is an ML workbench for exploring calibration of low-cost PM2.5 sensors. It compares raw sensor readings with a model trained against reference-monitor collocations, then evaluates that correction on a held-out station.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Start with the demo

The app generates a reproducible synthetic dataset with six stations. Choose one station as the test location and select **Run calibration**. AeroCal keeps one other station aside to calibrate the prediction interval, trains on the remaining stations, and reports results for the held-out location.

The synthetic readings include site-specific bias, humidity response, temperature response, and measurement noise. The true source values are generated only for this demo; they are not real air quality measurements.

## Use a collocation CSV

Upload a CSV with these required columns:

- `station_id`: identifier for the sensor/reference collocation site
- `sensor_pm25`: low-cost sensor reading in µg/m³
- `reference_pm25`: paired reference monitor reading in µg/m³
- `humidity`: relative humidity in percent
- `temperature`: degrees Celsius

An optional `timestamp` column is used for the time-series chart. AeroCal needs at least three distinct stations so it can keep separate sites for model fitting, interval calibration, and held-out evaluation.

## Method

A histogram gradient-boosting regressor learns a correction from sensor reading, humidity, and temperature. The split is by station: the model is fit on some sites, prediction intervals are calibrated on a separate site, and performance is reported on a held-out site.

The interval uses split-conformal residuals from the calibration station. Its nominal coverage relies on calibration and test errors being exchangeable; a new site or changing sensor behavior can break that assumption. The app reports observed held-out coverage so this limitation is visible.

## Scope and limits

AeroCal is an educational prototype, not a regulatory monitor or health advisory. Sensor calibration depends on sensor type, siting, environmental conditions, and reference data quality. Results from synthetic data do not validate a real sensor network.

For background, see the [US EPA air sensor FAQ](https://www.epa.gov/air-sensor-toolbox/frequent-questions-about-air-sensors) and the [OpenAQ API documentation](https://docs.openaq.org/about/about).

## License

The original project code is available under the MIT License.