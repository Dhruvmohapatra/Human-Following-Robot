import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PinConfig:
    # TB6612 #1 (Left Motors: Front Left & Rear Left)
    MOTOR1_PWMA = 12
    MOTOR1_AIN1 = 5
    MOTOR1_AIN2 = 6
    MOTOR1_PWMB = 13
    MOTOR1_BIN1 = 16
    MOTOR1_BIN2 = 20
    MOTOR1_STBY = 21

    # TB6612 #2 (Right Motors: Front Right & Rear Right)
    MOTOR2_PWMA = 18
    MOTOR2_AIN1 = 23
    MOTOR2_AIN2 = 24
    MOTOR2_PWMB = 19
    MOTOR2_BIN1 = 25
    MOTOR2_BIN2 = 26
    MOTOR2_STBY = 14

    # Front Ultrasonic Sensor
    FRONT_TRIG = 17
    FRONT_ECHO = 27

    # Rear Ultrasonic Sensor
    REAR_TRIG = 22
    REAR_ECHO = 4

class ControlConfig:
    # PD Controller for Steering
    STEERING_KP = 0.8
    STEERING_KD = 0.2
    
    # Proportional Controller for Speed
    SPEED_KP = 1.5
    
    # Target Following Thresholds
    # Bounding box height ratio relative to frame height (determines distance)
    DESIRED_HEIGHT_RATIO = 0.5   # Ideal height ratio (e.g. 50% of screen height)
    HEIGHT_DEADZONE = 0.05       # No speed change within +/- 5% of target height
    
    # Obstacle Avoidance thresholds
    OBSTACLE_MIN_DISTANCE_CM = 20.0  # Safe threshold to stop/reverse
    
    # Max speed scaling
    MAX_SPEED = 1.0        # Range: 0.0 to 1.0 (limits motor duty cycles)
    MIN_SPEED_THRESHOLD = 0.15  # Minimum duty cycle to overcome static friction
    
    # Timeout settings
    LOST_TARGET_TIMEOUT = 10.0  # Seconds to search before giving up

class CameraConfig:
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FRAME_RATE = 30
    
    # MobileNet SSD object detection
    MODEL_CONFIDENCE_THRESHOLD = 0.5
    MODEL_CLASS_PERSON_ID = 15  # Class 15 is person in COCO/Pascal VOC MobileNet SSD
    
    # Redundant detection check interval (to avoid tracker drift)
    REDETECTION_INTERVAL_FRAMES = 45  # Run full detection every N frames
    
    # Tracker type selection: "KCF", "CSRT", "MIL"
    TRACKER_TYPE = "KCF"

class SimConfig:
    # Simulator Window Settings
    WINDOW_WIDTH = 1024
    WINDOW_HEIGHT = 768
    
    # Physical to simulation conversions (pixels per meter)
    PIXELS_PER_METER = 100
    
    # Robot Physical Constants (Simulated)
    ROBOT_WIDTH_M = 0.25
    ROBOT_LENGTH_M = 0.35
    MAX_STEER_ANGLE_RAD = 0.5  # Max steering angle
    
    # Obstacle sensor simulation
    SENSOR_FOV_DEG = 30.0      # Ultrasonic sensor field of view
    SENSOR_MAX_RANGE_M = 4.0   # Max ultrasonic sensor range in meters
    
    # Default initial coordinates
    INITIAL_ROVER_X = 200
    INITIAL_ROVER_Y = 500
    INITIAL_ROVER_THETA = 0.0  # radians
    
    INITIAL_HUMAN_X = 500
    INITIAL_HUMAN_Y = 500
    
    # Interactive simulation key speed
    HUMAN_SPEED = 5.0          # Movement speed of human using keys

class PathConfig:
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    RUNS_DIR = os.path.join(BASE_DIR, "runs")
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    DOCS_DIR = os.path.join(BASE_DIR, "docs")
    
    # MobileNet SSD files
    MODEL_TXT = os.path.join(ASSETS_DIR, "MobileNetSSD_deploy.prototxt")
    MODEL_WEIGHTS = os.path.join(ASSETS_DIR, "MobileNetSSD_deploy.caffemodel")
    
    # CDNs to download model files if not present locally
    MODEL_TXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    MODEL_WEIGHTS_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

# Ensure directories exist
for p in [PathConfig.LOGS_DIR, PathConfig.RUNS_DIR, PathConfig.ASSETS_DIR, PathConfig.DOCS_DIR]:
    os.makedirs(p, exist_ok=True)

