import RPi.GPIO as GPIO
import time

# =========================
# RIGHT DRIVER
# =========================
AIN1_1 = 17
AIN2_1 = 27
PWMA1  = 18
STBY1  = 23

# =========================
# LEFT DRIVER
# =========================
AIN1_2 = 6
AIN2_2 = 13
PWMA2  = 12

BIN1_2 = 20
BIN2_2 = 21
PWMB2  = 16

STBY2  = 24

GPIO.setmode(GPIO.BCM)

pins = [
    AIN1_1, AIN2_1, PWMA1, STBY1,
    AIN1_2, AIN2_2, PWMA2,
    BIN1_2, BIN2_2, PWMB2, STBY2
]

for pin in pins:
    GPIO.setup(pin, GPIO.OUT)

GPIO.output(STBY1, GPIO.HIGH)
GPIO.output(STBY2, GPIO.HIGH)

# PWM
pwm_rightA = GPIO.PWM(PWMA1, 1000)
pwm_leftA  = GPIO.PWM(PWMA2, 1000)
pwm_leftB  = GPIO.PWM(PWMB2, 1000)

pwm_rightA.start(100)
pwm_leftA.start(100)
pwm_leftB.start(100)

# =========================
# RIGHT MOTOR A
# =========================
GPIO.output(AIN1_1, GPIO.HIGH)
GPIO.output(AIN2_1, GPIO.LOW)

# =========================
# LEFT MOTOR A
# =========================
GPIO.output(AIN1_2, GPIO.HIGH)
GPIO.output(AIN2_2, GPIO.LOW)

# =========================
# LEFT MOTOR B
# (reversed logic so it matches Left A)
# =========================
GPIO.output(BIN1_2, GPIO.LOW)
GPIO.output(BIN2_2, GPIO.HIGH)

print("RIGHT A + LEFT A + LEFT B RUNNING")

time.sleep(10)

pwm_rightA.stop()
pwm_leftA.stop()
pwm_leftB.stop()

GPIO.cleanup()

print("TEST COMPLETE")