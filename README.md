# Vision-Based Human Following Rover

[![Rover CI Pipeline](https://github.com/your-username/human_following_rover/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/human_following_rover/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A portfolio-grade robotics project implementing a **Vision-Based Human Following Rover** optimized for Raspberry Pi OS (64-bit). The project features a hybrid object tracking pipeline (MobileNet SSD + OpenCV Correlation Filters) and an interactive 2D physics/camera simulator allowing developers to test and iterate on control loops directly on Windows, macOS, or Linux without physical hardware.

---

## 🛠️ Hardware Requirements

* **SBC**: Raspberry Pi (Model 3B+, 4B, or 5 recommended) running Raspberry Pi OS (64-bit)
* **Camera**: Raspberry Pi Camera Module v1.3 (OV5647 5MP) or compatible
* **Motor Drivers**: 2x TB6612FNG Dual H-Bridge Motor Drivers (controlling 4 DC Motors)
* **Motors**: 4x DC Geared Motors (differential drive configuration)
* **Sensors**: 2x HC-SR04 Ultrasonic Sensors (Front and Rear safety bumpers)
* **Power Source**: 2S LiPo battery (7.4V) for motors, and a separate 5V power source for Raspberry Pi logic

---

## ⚙️ GPIO Mapping

Detailed wiring diagrams and protection circuitry guides are available in [docs/hardware_setup.md](docs/hardware_setup.md).

```
+-------------------------------------------------------------+
| RASPBERRY PI               TB6612 #1 (Left Motors)          |
| GPIO 12 (PWMA)     ---->   PWMA (Channel A Speed)           |
| GPIO 5  (AIN1)     ---->   AIN1 (Channel A Direction 1)     |
| GPIO 6  (AIN2)     ---->   AIN2 (Channel A Direction 2)     |
| GPIO 13 (PWMB)     ---->   PWMB (Channel B Speed)           |
| GPIO 16 (BIN1)     ---->   BIN1 (Channel B Direction 1)     |
| GPIO 20 (BIN2)     ---->   BIN2 (Channel B Direction 2)     |
| GPIO 21 (STBY)     ---->   STBY (Standby / Driver Enable)   |
+-------------------------------------------------------------+
| RASPBERRY PI               TB6612 #2 (Right Motors)         |
| GPIO 18 (PWMA)     ---->   PWMA (Channel A Speed)           |
| GPIO 23 (AIN1)     ---->   AIN1 (Channel A Direction 1)     |
| GPIO 24 (AIN2)     ---->   AIN2 (Channel A Direction 2)     |
| GPIO 19 (PWMB)     ---->   PWMB (Channel B Speed)           |
| GPIO 25 (BIN1)     ---->   BIN1 (Channel B Direction 1)     |
| GPIO 26 (BIN2)     ---->   BIN2 (Channel B Direction 2)     |
| GPIO 14 (STBY)     ---->   STBY (Standby / Driver Enable)   |
+-------------------------------------------------------------+
| RASPBERRY PI               ULTRASONIC SENSORS               |
| GPIO 17 / GPIO 27  ---->   TRIG / ECHO (Front Sensor)       |
| GPIO 22 / GPIO 4   ---->   TRIG / ECHO (Rear Sensor)        |
+-------------------------------------------------------------+
```

---

## 🚀 Key Features

* **Hybrid Tracking Pipeline**: Runs MobileNet SSD object detection to acquire target bounding boxes, then handsoff to OpenCV KCF tracking to achieve real-time (30 FPS) performance on low-power ARM CPUs, with periodic DNN drift correction.
* **Dual-Controller System**: Implements a Proportional-Derivative (PD) controller for smooth yaw/steering adjustments and a Proportional (P) speed controller to regulate distance based on bounding box size.
* **2D Simulation Environment**: Features a full-fledged simulator running side-by-side with the main loop. Includes keyboard/mouse target dragging, wall ray-casting for ultrasonic sensors, and virtual camera rendering.
* **Hardware-in-the-Loop Safe Override**: Active obstacle avoidance (front/rear) stops motors instantly if an obstacle is within 20cm, with hysteresis clearance thresholds (25cm) to prevent rapid state oscillation.
* **Telemetry and Session Recording**: Automatically saves CSV logs tracking FPS, latencies, CPU/Memory load, and motor speed history alongside composite AVI video captures of run overlays.
* **Multi-Platform CI**: GitHub Actions automated pipeline validating style guidelines and unit tests across Ubuntu, Windows, and macOS.

---

## 📐 Control Loop Mathematics

Detailed architecture and mathematics are documented in [docs/architecture.md](docs/architecture.md).

### PD Steering Controller
The steering error $e_{\text{steer}}(t)$ is computed from the bounding box horizontal offset and normalized:

$$e_{\text{steer}}(t) = \frac{x_{\text{target}} - x_{\text{center}}}{x_{\text{center}}}$$

$$u(t) = K_p \cdot e_{\text{steer}}(t) + K_d \cdot \frac{de_{\text{steer}}(t)}{dt}$$

### Proportional Speed Controller
Target distance error $e_{\text{speed}}(t)$ is derived from the target box height ratio:

$$e_{\text{speed}}(t) = r_{\text{desired}} - \frac{h_{\text{target}}}{H}$$

$$v(t) = K_{p\_speed} \cdot e_{\text{speed}}(t)$$

---

## 📁 Repository Structure

```
human_following_rover/
├── requirements.txt           # Dependency listings
├── config.py                 # Core configurations and pin assignments
├── logger.py                 # Double-target logger (console, files, CSV metrics)
├── main.py                   # Main loop & interactive 2D simulator
├── motor_controller.py       # TB6612FNG driver (Physical/Mock)
├── ultrasonic.py             # HC-SR04 ultrasonic interface & median filter
├── camera_tracker.py         # Threaded camera & vision pipeline
├── follow_controller.py      # PD controller & FSM state manager
├── tests/                    # Unit tests package
│   ├── test_motor_controller.py
│   ├── test_follow_controller.py
│   ├── test_state_machine.py
│   └── test_ultrasonic.py
├── docs/                     # Guides and architecture
└── .github/workflows/        # CI pipelines
```

---

## 🏁 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/human_following_rover.git
cd human_following_rover
pip install -r requirements.txt
```

### 2. Run in Simulation Mode (Windows/macOS/Linux)
On developers' computers, the code automatically detects the platform and starts simulation mode:
```bash
python main.py
```
* **Mouse Interactions**: Click and drag the green human target inside the Left Arena Panel to watch the rover steer and follow it.
* **Sensor Overrides**: Drag the human within 20cm of the front or rear sensor to trigger `BLOCKED` safety overrides.
* **Keyboards**: Press `q` or `ESC` inside the GUI window to stop the system cleanly.

### 3. Run on Physical Raspberry Pi
Once wired up, simply execute without arguments:
```bash
python main.py
```
The program will auto-download the MobileNet SSD weights on its first run and initialize physical GPIO control.

### 4. Run Test Suite
To verify the system's software consistency:
```bash
python -m unittest discover -s tests
```

---

## 📊 Telemetry Logging Format

CSV records saved inside `runs/` capture the following telemetry headers on every frame:
* `timestamp`: ISO-8601 timestamp.
* `state`: Current state machine state (SEARCHING, FOLLOWING, BLOCKED, etc.).
* `fps`: Loop cycle frequency.
* `detection_latency_ms`: CNN inference execution time (physical mode only).
* `tracking_latency_ms`: Correlation filter update time.
* `cpu_usage_pct` / `memory_usage_pct`: Local controller utilization metrics.
* `front_distance_cm` / `rear_distance_cm`: Smoothed ultrasonic readings.
* `motor_left_speed` / `motor_right_speed`: Active speed outputs mapped to duty cycles.
