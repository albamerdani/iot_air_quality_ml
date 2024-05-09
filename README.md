# IoT for Air Quality Monitoring and Data Analyze with Machine Learning models.

This is an IoT project for Air Quality Monitoring and analyze of collected data with Machine Learning models.

Dataset in different formats for data of: pm2.5, pm10, temperature, humidity, air pressure, alarm.

Python script for RasberryPi 3 and sensors configuration.

Runs on RaspberryPi 3 to measure and collect data through sensors - `air_quality_sensor.py`

Python and Jupyter scripts with ML models for data analyze for below use-cases.

ML Algorithms:
- Decission Tree
- Random Forest

Use cases:

- Predict alarm status for next n hours
- Predict future n values of alarm status
- Predict dust (or other parameters) based on historic samples in time-series
- Predict dust (or other parameters) future n values
- Decission Tree for alarm status based on all parameters historic values

## How to run

Clone the repo and install the necessary libraries:

`git clone https://github.com/albamerdani/iot_air_quality_ml.git
`
1. Install python3 and pip or pip3 - https://realpython.com/installing-python/
2. Install libraries under requirements.txt

`pip3 install -r requirements.txt`

3. Run python/jupyter scripts of different use-cases
