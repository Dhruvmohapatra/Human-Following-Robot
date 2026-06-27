import time
import logging
import statistics
from config import PinConfig, ControlConfig

logger = logging.getLogger("rover")

class MockDistanceSensor:
    """Mock class for HC-SR04 when gpiozero is not available."""
    def __init__(self, echo, trigger, max_distance=4.0):
        self.echo = echo
        self.trigger = trigger
        self.max_distance = max_distance
        # Default simulated distance is 1.5 meters (150 cm)
        self._distance = 1.5
        
    @property
    def distance(self):
        return self._distance
        
    @distance.setter
    def distance(self, val):
        self._distance = float(max(0.0, min(self.max_distance, val)))


class UltrasonicSensor:
    """Interfaces with a single HC-SR04 ultrasonic sensor with noise filtering."""
    def __init__(self, trigger_pin, echo_pin, label="Ultrasonic", sim_mode=False):
        self.label = label
        self.sim_mode = sim_mode
        self.max_distance_m = 4.0
        
        if sim_mode:
            self.sensor = MockDistanceSensor(echo_pin, trigger_pin, max_distance=self.max_distance_m)
        else:
            try:
                from gpiozero import DistanceSensor
                # max_distance is in meters
                self.sensor = DistanceSensor(
                    echo=echo_pin,
                    trigger=trigger_pin,
                    max_distance=self.max_distance_m,
                    queue_len=1 # We perform our own custom median filtering
                )
            except ImportError:
                logger.warning(
                    f"gpiozero not found. Falling back to simulated sensor for '{label}' (pins Trig:{trigger_pin}, Echo:{echo_pin})."
                )
                self.sim_mode = True
                self.sensor = MockDistanceSensor(echo_pin, trigger_pin, max_distance=self.max_distance_m)

    def read_distance_raw(self):
        """Reads raw distance from sensor and returns in centimeters."""
        try:
            # gpiozero returns distance in meters (0.0 to max_distance)
            dist_meters = self.sensor.distance
            # Convert to cm
            return dist_meters * 100.0
        except Exception as e:
            # In case of sensor hardware failure or timeout, return max distance to avoid blocking
            # unless we can confirm it is safe.
            return self.max_distance_m * 100.0

    def read_distance_filtered(self):
        """
        Takes a median of 3 reads to filter out high-frequency noise/spikes.
        Returns distance in centimeters.
        """
        reads = []
        for _ in range(3):
            reads.append(self.read_distance_raw())
            # Short sleep between pulses to prevent echoes from interfering with next measurement
            time.sleep(0.01)
            
        filtered = statistics.median(reads)
        return filtered

    def set_simulated_distance(self, distance_cm):
        """Allows simulation loop to inject distance measurements (in cm)."""
        if self.sim_mode:
            self.sensor.distance = distance_cm / 100.0

    def close(self):
        """Safely cleans up the sensor."""
        if not self.sim_mode:
            try:
                self.sensor.close()
            except Exception:
                pass

