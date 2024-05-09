import datetime
import smtplib
import sys
import time
import datetime
import requests
from email.mime.text import MIMEText

import Adafruit_BMP.BM085 as BMP085
import Adafruit_DHT
import RPi.GPIO as GPIO
import serial
from dotenv import dotenv_values

# Create sensor object for air pressure
sensor = BMP085.BMP085()

# Configure serial byte communication for USB ports for dust sensor
ser = serial.Serial('/dev/ttyUSB0')

# PIN Number for buzzer alarm, fan, data pin of temperature sensor
buzzerPin = 16
tempPin = 18
fanPin = 8
fanMinSpeed = 25
PWM_FREQ = 25
GPIO.setwarnings(False)

# Mode of PIN Number
GPIO.setmode(GPIO.BOARD)

# Set alarm pin as output
GPIO.setup(buzzerPin, GPIO.OUT)

# Set temperature sensor pin as input
# Set fan sensor pin as output and frequence
GPIO.setup(tempPin, GPIO.IN)
GPIO.setup(fanPin, GPIO.OUT)
fan = GPIO.PWM(fanPin, PWM_FREQ)
fan.start(0)

x = 0
alarm = 0  # False

# Load environment variables from .env file
env_vars = dotenv_values(".env")

# Write API KEY
myAPI = env_vars["API_KEY"]
# URL of thingspeak cloud platform
baseURL = env_vars["BASE_URL"] % myAPI

def fan_cycle(temp):
    """
    Start fan if temperature is above a certain value
    :param temp:
    :return: change fan status
    """
    if temp > 25:
        fan.ChangeDutyCycle(25)
        print("Fan open!")
    else:
        fan.ChangeDutyCycle(0)
        print("Fan close!")
        fan.stop()


# Send sensor values in Cloud Platform
def sendData_Cloud(pm2_5, pm10, temp, pressure, alarm):

    # Define the parameters
    parameters = {
        'field1': pm2_5,
        'field2': pm10,
        'field3': temp,
        'field4': pressure,
        'field5': alarm
    }

    # Make the POST request
    response = requests.post(baseURL, data=parameters)

    # Check if the request was successful
    if response.status_code == 200:
        print("Data successfully sent to cloud platform.")
    else:
        print("Error:", response.text)


def store_value_file(data_time, pm2_5, pm10, temp, trysni, alarm_status):
    path_file = os.path.join("/", "parametra_ajri.csv")
    with open(path_file, 'a') as log:
        log.write(
            "{0},{1},{2},{3},{4},{5}\n".format(data_time,str(pm2_5),str(pm10),str(temp),str(trysni),str(alarm_status)))


# send email notification to user
def send_notification():

    USERNAME = env_vars["USERNAME"]
    PASSWORD = env_vars["PASSWORD"]
    MAILTO = env_vars["MAILTO"]

    msg = MIMEText('Alarm is on. Environment parameters are not normal.')
    msg['Subject'] = 'IoT Notification'
    msg['From'] = USERNAME
    msg['To'] = MAILTO

    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo_or_helo_if_needed()
    server.starttls()
    server.ehlo_or_helo_if_needed()
    server.login(USERNAME, PASSWORD)
    server.sendmail(USERNAME, MAILTO, msg.as_string())
    server.quit()


def main(cycles: int):
    while x < cycles:
        data = []
        for index in range(0, 10):
            datum = ser.read()
            data.append(datum)

        data_time = datetime.datetime.now()
        pm2_5 = int.from_bytes(b''.join(data[2:4]), byteorder='little')/10
        pm10 = int.from_bytes(b''.join(data[4:6]), byteorder='little')/10

        temp = sensor.read_temperature()
        pressure = sensor.read_pressure()

        humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT11, tempPin)

        print("Date and time:" + str(data_time))
        print("PM 2.5: " + str(pm2_5))
        print("PM 10: " + str(pm10))
        print('Humidity = ', humidity)
        print ('Temperature = {0:0.2f} *C'.format(temp))                # temperature Celcius
        print ('Air Pressure = {0:0.2f} Pa'.format(pressure))           # Air pressure
        print ('Altitude = {0:0.2f} m'.format(sensor.read_altitude()))  # Current altitude
        print ('Air Pressure in sea level = {0:0.2f} Pa'.format(sensor.read_sealevel_pressure())) # Air pressure in sea level

        fan_cycle(temp)

        if pm2_5 > 14 or pm10 > 25 or temp > 25 or temp < 18:
            GPIO.output(buzzerPin, GPIO.HIGH)       #Alarm generated - set to True if parameters are above or below certain values
            alarm = 1                               #True
            send_notification()
            print('Alarm!! Beep!')
            time.sleep(2)                          # beeps for 2 sec
            GPIO.output(buzzerPin, GPIO.LOW)
            time.sleep(1)                           # sleep for 1 sec
        else:
            GPIO.output(buzzerPin, GPIO.LOW)
            alarm = 0
            print('No alarm')

        sendData_Cloud(pm2_5, pm10, temp, humidity, pressure, alarm)
        store_value_file(data_time, pm2_5, pm10, temp, humidity, pressure, alarm)

        x = x + 1
        print('')
        time.sleep(15)


    GPIO.cleanup()
    sys.exit(1)


if __name__ == 'main':
    main(cycles=1000)
