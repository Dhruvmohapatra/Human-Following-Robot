import logging
from config import PinConfig

logger = logging.getLogger("rover")

# Mock classes for Simulation Mode or when running on non-RPi platforms
class MockDigitalOutputDevice:
    def __init__(self, pin):
        self.pin = pin
        self._value = 0
        
    @property
    def value(self):
        return self._value
        
    @value.setter
    def value(self, val):
        self._value = int(val)
        
    def on(self):
        self.value = 1
        
    def off(self):
        self.value = 0

class MockPWMOutputDevice:
    def __init__(self, pin, frequency=100):
        self.pin = pin
        self.frequency = frequency
        self._value = 0.0
        
    @property
    def value(self):
        return self._value
        
    @value.setter
    def value(self, val):
        self._value = float(max(0.0, min(1.0, val)))
        
    def close(self):
        pass


class TB6612Channel:
    """Controls a single channel (A or B) of a TB6612FNG motor driver."""
    def __init__(self, pwm_pin, in1_pin, in2_pin, sim_mode=False):
        self.sim_mode = sim_mode
        self._left_right_label = "" # Used for diagnostic logging
        
        if sim_mode:
            self.pwm = MockPWMOutputDevice(pwm_pin)
            self.in1 = MockDigitalOutputDevice(in1_pin)
            self.in2 = MockDigitalOutputDevice(in2_pin)
        else:
            try:
                from gpiozero import PWMOutputDevice, DigitalOutputDevice
                # Frequency set to 100Hz which is suitable for standard DC motors
                self.pwm = PWMOutputDevice(pwm_pin, frequency=100)
                self.in1 = DigitalOutputDevice(in1_pin)
                self.in2 = DigitalOutputDevice(in2_pin)
            except ImportError:
                logger.warning(
                    f"gpiozero not found. Falling back to simulated channel for pins ({pwm_pin}, {in1_pin}, {in2_pin})."
                )
                self.sim_mode = True
                self.pwm = MockPWMOutputDevice(pwm_pin)
                self.in1 = MockDigitalOutputDevice(in1_pin)
                self.in2 = MockDigitalOutputDevice(in2_pin)

    def set_speed(self, speed):
        """
        Sets speed of the motor.
        speed: float from -1.0 (full reverse) to 1.0 (full forward)
        """
        # Clamp speed
        speed = max(-1.0, min(1.0, speed))
        
        if speed > 0.0:
            # Forward: IN1 High, IN2 Low
            self.in1.value = 1
            self.in2.value = 0
            self.pwm.value = speed
        elif speed < 0.0:
            # Reverse: IN1 Low, IN2 High
            self.in1.value = 0
            self.in2.value = 1
            self.pwm.value = abs(speed)
        else:
            # Brake: IN1 Low, IN2 Low (or High/High for active braking; Low/Low is coast/stop)
            self.in1.value = 0
            self.in2.value = 0
            self.pwm.value = 0.0

    def get_speed(self):
        """Returns the current speed and direction (signed float)."""
        val = self.pwm.value
        if self.in1.value == 1 and self.in2.value == 0:
            return val
        elif self.in1.value == 0 and self.in2.value == 1:
            return -val
        return 0.0

    def close(self):
        if not self.sim_mode:
            try:
                self.pwm.close()
            except Exception:
                pass


class MotorController:
    """Manages the rover's 4 motors using two TB6612FNG motor drivers."""
    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        
        logger.info(f"Initializing Motor Controller (Simulation Mode: {sim_mode})")
        
        # Initialize Standby pins
        if self.sim_mode:
            self.stby1 = MockDigitalOutputDevice(PinConfig.MOTOR1_STBY)
            self.stby2 = MockDigitalOutputDevice(PinConfig.MOTOR2_STBY)
        else:
            try:
                from gpiozero import DigitalOutputDevice
                self.stby1 = DigitalOutputDevice(PinConfig.MOTOR1_STBY)
                self.stby2 = DigitalOutputDevice(PinConfig.MOTOR2_STBY)
            except ImportError:
                self.sim_mode = True
                self.stby1 = MockDigitalOutputDevice(PinConfig.MOTOR1_STBY)
                self.stby2 = MockDigitalOutputDevice(PinConfig.MOTOR2_STBY)

        # Initialize Motor Channels
        # TB6612 #1 (Left Front + Left Rear)
        self.left_front = TB6612Channel(
            PinConfig.MOTOR1_PWMA, PinConfig.MOTOR1_AIN1, PinConfig.MOTOR1_AIN2, self.sim_mode
        )
        self.left_rear = TB6612Channel(
            PinConfig.MOTOR1_PWMB, PinConfig.MOTOR1_BIN1, PinConfig.MOTOR1_BIN2, self.sim_mode
        )
        
        # TB6612 #2 (Right Front + Right Rear)
        self.right_front = TB6612Channel(
            PinConfig.MOTOR2_PWMA, PinConfig.MOTOR2_AIN1, PinConfig.MOTOR2_AIN2, self.sim_mode
        )
        self.right_rear = TB6612Channel(
            PinConfig.MOTOR2_PWMB, PinConfig.MOTOR2_BIN1, PinConfig.MOTOR2_BIN2, self.sim_mode
        )

        # Labels for debugging
        self.left_front._left_right_label = "Left Front"
        self.left_rear._left_right_label = "Left Rear"
        self.right_front._left_right_label = "Right Front"
        self.right_rear._left_right_label = "Right Rear"
        
        # Enable motor drivers (STBY pins must be HIGH)
        self.enable_drivers()

    def enable_drivers(self):
        """Enables the TB6612FNG drivers by pulling STBY pins HIGH."""
        self.stby1.value = 1
        self.stby2.value = 1

    def disable_drivers(self):
        """Disables the TB6612FNG drivers by pulling STBY pins LOW."""
        self.stby1.value = 0
        self.stby2.value = 0

    def set_speeds(self, left_speed, right_speed):
        """
        Sets the speeds of left and right motor pairs.
        left_speed: Float from -1.0 to 1.0
        right_speed: Float from -1.0 to 1.0
        """
        # Apply speed settings to channels
        self.left_front.set_speed(left_speed)
        self.left_rear.set_speed(left_speed)
        
        self.right_front.set_speed(right_speed)
        self.right_rear.set_speed(right_speed)

    def get_speeds(self):
        """Returns current average speed of left and right motors: (left_speed, right_speed)."""
        left_avg = (self.left_front.get_speed() + self.left_rear.get_speed()) / 2.0
        right_avg = (self.right_front.get_speed() + self.right_rear.get_speed()) / 2.0
        return left_avg, right_avg

    def stop(self):
        """Brakes/stops all motors."""
        self.set_speeds(0.0, 0.0)

    def cleanup(self):
        """Turns off drivers and releases GPIO resources."""
        logger.info("Cleaning up Motor Controller resources")
        self.stop()
        self.disable_drivers()
        self.left_front.close()
        self.left_rear.close()
        self.right_front.close()
        self.right_rear.close()

