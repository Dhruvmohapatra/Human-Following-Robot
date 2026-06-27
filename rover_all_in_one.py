#!/usr/bin/env python3
"""
All-in-one human-following rover script with front/rear obstacle stop.

Run on the Raspberry Pi with system Python:
    cd /home/pi/human_following_rover
    /usr/bin/python3 rover_all_in_one.py --display

Headless SSH run:
    /usr/bin/python3 rover_all_in_one.py
"""

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass

# Raspberry Pi OS installs Picamera2/lgpio in the system dist-packages path.
# Prefer that path so Picamera2 uses the system NumPy ABI it was built against.
SYSTEM_DIST_PACKAGES = "/usr/lib/python3/dist-packages"
if os.path.isdir(SYSTEM_DIST_PACKAGES) and SYSTEM_DIST_PACKAGES not in sys.path:
    sys.path.insert(0, SYSTEM_DIST_PACKAGES)

import cv2
import numpy as np


# -----------------------------
# GPIO / hardware configuration
# -----------------------------

@dataclass(frozen=True)
class Pins:
    # TB6612 #1: left front and left rear motors, BCM GPIO numbering.
    MOTOR1_PWMA: int = 12
    MOTOR1_AIN1: int = 5
    MOTOR1_AIN2: int = 6
    MOTOR1_PWMB: int = 13
    MOTOR1_BIN1: int = 16
    MOTOR1_BIN2: int = 20
    MOTOR1_STBY: int = 21

    # TB6612 #2: right front and right rear motors, BCM GPIO numbering.
    MOTOR2_PWMA: int = 18
    MOTOR2_AIN1: int = 23
    MOTOR2_AIN2: int = 24
    MOTOR2_PWMB: int = 19
    MOTOR2_BIN1: int = 25
    MOTOR2_BIN2: int = 26
    MOTOR2_STBY: int = 14

    # HC-SR04 sensors, BCM GPIO numbering.
    FRONT_TRIG: int = 17
    FRONT_ECHO: int = 27
    REAR_TRIG: int = 22
    REAR_ECHO: int = 4


PINS = Pins()


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TXT = os.path.join(BASE_DIR, "assets", "MobileNetSSD_deploy.prototxt")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "assets", "MobileNetSSD_deploy.caffemodel")

PERSON_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.45

TARGET_HEIGHT_RATIO = 0.40
HEIGHT_DEADZONE = 0.05
MAX_SPEED = 0.45
MIN_SPEED = 0.18
STEERING_KP = 0.45
STEERING_KD = 0.04
SPEED_KP = 1.2

FRONT_STOP_CM = 18.0
REAR_STOP_CM = 18.0
SEARCH_SPEED = 0.22
LOST_TARGET_TIMEOUT_SEC = 8.0


class MotorChannel:
    def __init__(self, pwm_pin, in1_pin, in2_pin, pwm_freq=100, inverted=False):
        from gpiozero import DigitalOutputDevice, PWMOutputDevice

        self.inverted = inverted
        self.pwm = PWMOutputDevice(pwm_pin, frequency=pwm_freq)
        self.in1 = DigitalOutputDevice(in1_pin)
        self.in2 = DigitalOutputDevice(in2_pin)

    def set_speed(self, speed):
        speed = max(-1.0, min(1.0, float(speed)))
        if self.inverted:
            speed = -speed
        if speed > 0:
            self.in1.on()
            self.in2.off()
            self.pwm.value = speed
        elif speed < 0:
            self.in1.off()
            self.in2.on()
            self.pwm.value = abs(speed)
        else:
            self.in1.off()
            self.in2.off()
            self.pwm.value = 0.0

    def close(self):
        self.set_speed(0.0)
        self.pwm.close()
        self.in1.close()
        self.in2.close()


class MotorController:
    def __init__(self):
        from gpiozero import DigitalOutputDevice

        self.stby1 = DigitalOutputDevice(PINS.MOTOR1_STBY)
        self.stby2 = DigitalOutputDevice(PINS.MOTOR2_STBY)

        # Motor A is physically reversed on this rover. This matches the user's
        # verified RPi.GPIO test: A forward uses AIN1 LOW, AIN2 HIGH.
        self.left_front = MotorChannel(PINS.MOTOR1_PWMA, PINS.MOTOR1_AIN1, PINS.MOTOR1_AIN2, inverted=True)
        self.left_rear = MotorChannel(PINS.MOTOR1_PWMB, PINS.MOTOR1_BIN1, PINS.MOTOR1_BIN2)
        self.right_front = MotorChannel(PINS.MOTOR2_PWMA, PINS.MOTOR2_AIN1, PINS.MOTOR2_AIN2)
        self.right_rear = MotorChannel(PINS.MOTOR2_PWMB, PINS.MOTOR2_BIN1, PINS.MOTOR2_BIN2)

        self.enable()
        self.stop()

    def enable(self):
        self.stby1.on()
        self.stby2.on()

    def disable(self):
        self.stby1.off()
        self.stby2.off()

    def set_speeds(self, left_speed, right_speed):
        left_speed = self._apply_min_speed(left_speed)
        right_speed = self._apply_min_speed(right_speed)

        self.left_front.set_speed(left_speed)
        self.left_rear.set_speed(left_speed)
        self.right_front.set_speed(right_speed)
        self.right_rear.set_speed(right_speed)

    def stop(self):
        self.left_front.set_speed(0.0)
        self.left_rear.set_speed(0.0)
        self.right_front.set_speed(0.0)
        self.right_rear.set_speed(0.0)

    def close(self):
        self.stop()
        self.disable()
        self.left_front.close()
        self.left_rear.close()
        self.right_front.close()
        self.right_rear.close()
        self.stby1.close()
        self.stby2.close()

    @staticmethod
    def _apply_min_speed(speed):
        if abs(speed) < 0.001:
            return 0.0
        if abs(speed) < MIN_SPEED:
            return MIN_SPEED if speed > 0 else -MIN_SPEED
        return max(-1.0, min(1.0, speed))


class UltrasonicPair:
    def __init__(self):
        import RPi.GPIO as GPIO

        self.GPIO = GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PINS.FRONT_TRIG, GPIO.OUT)
        GPIO.setup(PINS.FRONT_ECHO, GPIO.IN)
        GPIO.setup(PINS.REAR_TRIG, GPIO.OUT)
        GPIO.setup(PINS.REAR_ECHO, GPIO.IN)
        GPIO.output(PINS.FRONT_TRIG, GPIO.LOW)
        GPIO.output(PINS.REAR_TRIG, GPIO.LOW)
        time.sleep(0.05)

    def read_cm(self):
        front_cm = self._read_one(PINS.FRONT_TRIG, PINS.FRONT_ECHO)
        time.sleep(0.01)
        rear_cm = self._read_one(PINS.REAR_TRIG, PINS.REAR_ECHO)
        return front_cm, rear_cm

    def _read_one(self, trig_pin, echo_pin):
        reads = [self._pulse_distance_cm(trig_pin, echo_pin) for _ in range(2)]
        return min(reads)

    def _pulse_distance_cm(self, trig_pin, echo_pin, timeout_sec=0.025):
        try:
            GPIO = self.GPIO
            GPIO.output(trig_pin, GPIO.LOW)
            time.sleep(0.00002)
            GPIO.output(trig_pin, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(trig_pin, GPIO.LOW)

            start_deadline = time.monotonic() + timeout_sec
            while GPIO.input(echo_pin) == GPIO.LOW:
                if time.monotonic() > start_deadline:
                    return 400.0
            pulse_start = time.monotonic()

            end_deadline = pulse_start + timeout_sec
            while GPIO.input(echo_pin) == GPIO.HIGH:
                if time.monotonic() > end_deadline:
                    return 400.0
            pulse_end = time.monotonic()

            return min(400.0, (pulse_end - pulse_start) * 17150.0)
        except Exception:
            return 400.0

    def close(self):
        try:
            self.GPIO.cleanup([PINS.FRONT_TRIG, PINS.FRONT_ECHO, PINS.REAR_TRIG, PINS.REAR_ECHO])
        except Exception:
            pass


class Camera:
    def __init__(self):
        self.picam = None
        self.cap = None

        try:
            from picamera2 import Picamera2

            self.picam = Picamera2()
            config = self.picam.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
            )
            self.picam.configure(config)
            self.picam.start()
            time.sleep(1.0)
            print("Camera: Picamera2 active")
            return
        except Exception as exc:
            print(f"Camera: Picamera2 unavailable ({exc}); trying OpenCV camera 0")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        ok, _ = self.cap.read()
        if not ok:
            raise RuntimeError("No camera frame available from Picamera2 or OpenCV VideoCapture(0)")
        print("Camera: OpenCV VideoCapture active")

    def read(self):
        if self.picam:
            frame = self.picam.capture_array()
            if frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.picam:
            self.picam.stop()
            self.picam.close()
        if self.cap:
            self.cap.release()


class PersonDetector:
    def __init__(self):
        if not os.path.exists(MODEL_TXT) or not os.path.exists(MODEL_WEIGHTS):
            raise FileNotFoundError(
                "MobileNet SSD model files are missing. Expected assets/"
                "MobileNetSSD_deploy.prototxt and assets/MobileNetSSD_deploy.caffemodel"
            )
        self.net = cv2.dnn.readNetFromCaffe(MODEL_TXT, MODEL_WEIGHTS)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_person(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        best_box = None
        best_conf = 0.0
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            class_id = int(detections[0, 0, i, 1])
            if class_id != PERSON_CLASS_ID or conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, box_w, box_h)

        return best_box, best_conf


class FollowBrain:
    def __init__(self):
        self.prev_steer_error = 0.0
        self.last_seen_time = 0.0
        self.last_known_error = 0.0

    def compute(self, box, front_cm, rear_cm, dt):
        front_blocked = front_cm < FRONT_STOP_CM
        rear_blocked = rear_cm < REAR_STOP_CM

        if front_blocked and rear_blocked:
            return 0.0, 0.0, "BLOCKED_BOTH"

        if box is None:
            if time.time() - self.last_seen_time < LOST_TARGET_TIMEOUT_SEC:
                spin = SEARCH_SPEED if self.last_known_error >= 0 else -SEARCH_SPEED
                return -spin, spin, "LOST_SEARCH"
            return -SEARCH_SPEED, SEARCH_SPEED, "SEARCHING"

        self.last_seen_time = time.time()
        x, y, w, h = box

        target_center_x = x + w / 2.0
        frame_center_x = FRAME_WIDTH / 2.0
        steer_error = (target_center_x - frame_center_x) / frame_center_x
        self.last_known_error = steer_error

        steering = STEERING_KP * steer_error + STEERING_KD * ((steer_error - self.prev_steer_error) / max(dt, 0.02))
        steering = max(-0.35, min(0.35, steering))
        self.prev_steer_error = steer_error

        height_ratio = h / float(FRAME_HEIGHT)
        speed_error = TARGET_HEIGHT_RATIO - height_ratio
        if abs(speed_error) < HEIGHT_DEADZONE:
            forward = 0.0
        else:
            forward = max(-MAX_SPEED, min(MAX_SPEED, SPEED_KP * speed_error))

        if forward > 0 and front_blocked:
            forward = 0.0
            state = "FRONT_OBSTACLE_STOP"
        elif forward < 0 and rear_blocked:
            forward = 0.0
            state = "REAR_OBSTACLE_STOP"
        else:
            state = "FOLLOWING"

        left = max(-MAX_SPEED, min(MAX_SPEED, forward + steering))
        right = max(-MAX_SPEED, min(MAX_SPEED, forward - steering))
        return left, right, state


def draw_overlay(frame, box, conf, state, front_cm, rear_cm, left, right):
    if box:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"person {conf:.2f}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    lines = [
        f"STATE: {state}",
        f"FRONT: {front_cm:.0f} cm  REAR: {rear_cm:.0f} cm",
        f"MOTORS L:{left:.2f} R:{right:.2f}",
        "q/ESC quits",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 24
    return frame


def describe_motion(left, right):
    avg = (left + right) / 2.0
    turn = right - left
    if abs(left) < 0.01 and abs(right) < 0.01:
        return "STOP"
    if avg > 0.08:
        base = "MOVE_FORWARD"
    elif avg < -0.08:
        base = "MOVE_BACKWARD"
    elif turn > 0:
        base = "SEARCH_OR_TURN_RIGHT"
    else:
        base = "SEARCH_OR_TURN_LEFT"
    if abs(turn) > 0.12 and abs(avg) > 0.08:
        base += "_WITH_STEER"
    return base


def main():
    parser = argparse.ArgumentParser(description="Human-following rover with obstacle avoidance.")
    parser.add_argument("--display", action="store_true", help="Show live OpenCV window. Use only with a desktop/VNC display.")
    parser.add_argument("--dry-run", action="store_true", help="Run camera/detection/sensors but do not move motors.")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Stop cleanly after this many seconds. 0 means run until Ctrl+C.")
    args = parser.parse_args()

    running = True

    def handle_stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    motors = None
    sensors = None
    camera = None
    try:
        print("Starting rover_all_in_one.py")
        print("Recommended run command: venv/bin/python rover_all_in_one.py")

        camera = Camera()
        detector = PersonDetector()
        brain = FollowBrain()
        sensors = UltrasonicPair()
        motors = None if args.dry_run else MotorController()

        last_time = time.time()
        start_time = last_time
        frame_count = 0
        last_box = None
        last_conf = 0.0
        last_report_key = None

        while running:
            if args.max_seconds > 0 and time.time() - start_time >= args.max_seconds:
                print("Max runtime reached.")
                running = False
                break

            now = time.time()
            dt = now - last_time
            last_time = now

            frame = camera.read()
            if frame is None:
                print("WARN: empty camera frame")
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            front_cm, rear_cm = sensors.read_cm()

            # Detect every frame for reliability. If CPU is too slow, change this to every 2-3 frames.
            box, conf = detector.detect_person(frame)
            if box is not None:
                last_box, last_conf = box, conf
            else:
                last_box, last_conf = None, 0.0

            left, right, state = brain.compute(last_box, front_cm, rear_cm, dt)
            if motors:
                motors.set_speeds(left, right)

            frame_count += 1
            person_status = "PERSON_DETECTED" if last_box is not None else "NO_PERSON"
            front_status = "FRONT_OBSTACLE" if front_cm < FRONT_STOP_CM else "FRONT_CLEAR"
            rear_status = "REAR_OBSTACLE" if rear_cm < REAR_STOP_CM else "REAR_CLEAR"
            motion = describe_motion(left, right)
            if last_box is not None:
                bx, by, bw, bh = last_box
                box_msg = f"box=({bx},{by},{bw},{bh}) height_ratio={bh / float(FRAME_HEIGHT):.2f}"
            else:
                box_msg = "box=None height_ratio=0.00"

            report_key = (person_status, front_status, rear_status, motion, state)
            should_report_event = report_key != last_report_key
            if should_report_event:
                last_report_key = report_key

            if should_report_event or frame_count % 5 == 0:
                print(
                    f"{state:20s} {person_status:15s} {motion:24s} "
                    f"{front_status:14s} {rear_status:13s} "
                    f"front={front_cm:6.1f}cm rear={rear_cm:6.1f}cm "
                    f"conf={last_conf:.2f} L={left:+.2f} R={right:+.2f} {box_msg}"
                )

            if args.display:
                display = draw_overlay(frame.copy(), last_box, last_conf, state, front_cm, rear_cm, left, right)
                cv2.imshow("Human Following Rover", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    running = False

        print("Stopping rover...")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if motors:
            motors.close()
        if sensors:
            sensors.close()
        if camera:
            camera.close()
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
