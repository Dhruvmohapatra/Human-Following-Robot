import time
import logging
from config import ControlConfig, CameraConfig

logger = logging.getLogger("rover")

class RoverState:
    INITIALIZING = "INITIALIZING"
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    FOLLOWING = "FOLLOWING"
    BLOCKED = "BLOCKED"
    LOST_TARGET = "LOST_TARGET"
    SHUTDOWN = "SHUTDOWN"


class FollowController:
    """Manages the rover's control loop (PD/P controllers) and State Machine transitions."""
    def __init__(self):
        # PD steering controller states
        self.kp_steer = ControlConfig.STEERING_KP
        self.kd_steer = ControlConfig.STEERING_KD
        self.prev_steer_error = 0.0
        
        # Speed controller states
        self.kp_speed = ControlConfig.SPEED_KP
        self.desired_height_ratio = ControlConfig.DESIRED_HEIGHT_RATIO
        
        # State machine variables
        self.state = RoverState.INITIALIZING
        self.last_state = RoverState.INITIALIZING
        
        # Tracking history
        self.target_lost_time = 0.0
        self.last_known_error = 0.0  # Positive = target was on right, Negative = left
        self.tracking_confirm_frames = 0
        self.required_confirm_frames = 5  # Frames to confirm lock before following

    def change_state(self, new_state):
        """Changes the state and logs the transition."""
        if self.state != new_state:
            logger.info(f"FSM State Transition: {self.state} -> {new_state}")
            self.last_state = self.state
            self.state = new_state
            
            # Reset state-specific variables on transition
            if new_state == RoverState.LOST_TARGET:
                self.target_lost_time = time.time()
            elif new_state == RoverState.TRACKING:
                self.tracking_confirm_frames = 0
            elif new_state == RoverState.SEARCHING:
                self.prev_steer_error = 0.0

    def update(self, tracking_box, front_distance, rear_distance, dt):
        """
        Executes one control loop step.
        tracking_box: (x, y, w, h) of target, or None if lost
        front_distance: Distance in cm to front obstacle
        rear_distance: Distance in cm to rear obstacle
        dt: Elapsed time in seconds since last update
        
        Returns: (left_motor_speed, right_motor_speed) as signed floats in [-1.0, 1.0]
        """
        # Ensure dt is safe
        if dt <= 0.0:
            dt = 0.033  # Default to ~30 FPS time step if invalid
            
        # Check safety overrides first
        # Hysteresis: We stay blocked if we are already blocked and distance < 25.
        # If not blocked, we trigger if distance < 20.
        is_front_blocked = (front_distance < ControlConfig.OBSTACLE_MIN_DISTANCE_CM) if self.state != RoverState.BLOCKED else (front_distance < 25.0)
        is_rear_blocked = (rear_distance < ControlConfig.OBSTACLE_MIN_DISTANCE_CM) if self.state != RoverState.BLOCKED else (rear_distance < 25.0)

        # ----------------------------------------------------
        # STATE MACHINE TRANSITIONS
        # ----------------------------------------------------
        
        # Transition out of INITIALIZING
        if self.state == RoverState.INITIALIZING:
            self.change_state(RoverState.SEARCHING)
            
        # Handle SHUTDOWN
        elif self.state == RoverState.SHUTDOWN:
            return 0.0, 0.0

        # Handle BLOCKED state
        elif self.state == RoverState.BLOCKED:
            if not is_front_blocked and not is_rear_blocked:
                # Obstacles cleared, return to searching
                self.change_state(RoverState.SEARCHING)
            else:
                return 0.0, 0.0

        # Handle target lost transitions
        if tracking_box is None:
            if self.state in [RoverState.TRACKING, RoverState.FOLLOWING]:
                self.change_state(RoverState.LOST_TARGET)
        else:
            if self.state in [RoverState.SEARCHING, RoverState.LOST_TARGET]:
                self.change_state(RoverState.TRACKING)

        # State-specific actions and transitions
        if self.state == RoverState.SEARCHING:
            # Check if person was detected (would have transitioned to TRACKING above)
            # Otherwise, spin slowly in place to search
            left_speed = -0.3
            right_speed = 0.3
            return left_speed, right_speed

        elif self.state == RoverState.LOST_TARGET:
            # Spin in last known direction of the target
            spin_dir = 1.0 if self.last_known_error >= 0.0 else -1.0
            left_speed = 0.3 * spin_dir
            right_speed = -0.3 * spin_dir
            
            # Check timeout
            elapsed = time.time() - self.target_lost_time
            if elapsed > ControlConfig.LOST_TARGET_TIMEOUT:
                logger.warning(f"Target lost for {ControlConfig.LOST_TARGET_TIMEOUT}s. Returning to SEARCHING.")
                self.change_state(RoverState.SEARCHING)
                
            return left_speed, right_speed

        elif self.state == RoverState.TRACKING:
            # Target is present, stop motors and confirm lock
            self.tracking_confirm_frames += 1
            if self.tracking_confirm_frames >= self.required_confirm_frames:
                self.change_state(RoverState.FOLLOWING)
            return 0.0, 0.0

        elif self.state == RoverState.FOLLOWING:
            # Evaluate if we are attempting to move into an obstacle
            x, y, w, h = tracking_box
            current_height_ratio = h / float(CameraConfig.FRAME_HEIGHT)
            speed_error = self.desired_height_ratio - current_height_ratio
            
            if speed_error > ControlConfig.HEIGHT_DEADZONE and is_front_blocked:
                self.change_state(RoverState.BLOCKED)
                return 0.0, 0.0
            if speed_error < -ControlConfig.HEIGHT_DEADZONE and is_rear_blocked:
                self.change_state(RoverState.BLOCKED)
                return 0.0, 0.0

            # Calculate control outputs
            left_speed, right_speed = self._compute_follow_speeds(
                tracking_box, is_front_blocked, is_rear_blocked, dt
            )
            return left_speed, right_speed

        return 0.0, 0.0

    def _compute_follow_speeds(self, tracking_box, is_front_blocked, is_rear_blocked, dt):
        """Computes differential motor speeds using PD (steering) and Proportional (speed)."""
        x, y, w, h = tracking_box
        
        # 1. Steering Error calculation (heading offset)
        # Normalise error to [-1.0, 1.0] where 0 is center
        frame_width = CameraConfig.FRAME_WIDTH
        target_center_x = x + w / 2.0
        frame_center_x = frame_width / 2.0
        
        steer_error = (target_center_x - frame_center_x) / frame_center_x
        self.last_known_error = steer_error
        
        # PD Steering Control
        p_term = self.kp_steer * steer_error
        d_term = self.kd_steer * (steer_error - self.prev_steer_error) / dt
        steering_output = p_term + d_term
        self.prev_steer_error = steer_error
        
        # 2. Speed Error calculation (distance estimation based on height ratio)
        frame_height = CameraConfig.FRAME_HEIGHT
        current_height_ratio = h / float(frame_height)
        
        # Positive error = too far away (speed forward), Negative = too close (speed backward)
        speed_error = self.desired_height_ratio - current_height_ratio
        
        # Proportional Speed Control
        if abs(speed_error) < ControlConfig.HEIGHT_DEADZONE:
            speed_output = 0.0
        else:
            speed_output = self.kp_speed * speed_error
            
        # 3. Handle Obstacle constraints on speed output
        if speed_output > 0.0 and is_front_blocked:
            speed_output = 0.0
        if speed_output < 0.0 and is_rear_blocked:
            speed_output = 0.0

        # 4. Mix steering and speed using differential drive equations
        left_speed = speed_output + steering_output
        right_speed = speed_output - steering_output

        # 5. Apply static friction correction & scaling
        left_speed = self._apply_friction_correction(left_speed)
        right_speed = self._apply_friction_correction(right_speed)
        
        # Scale to maximum allowed config speed
        left_speed *= ControlConfig.MAX_SPEED
        right_speed *= ControlConfig.MAX_SPEED
        
        # Final safety clamp
        left_speed = max(-1.0, min(1.0, left_speed))
        right_speed = max(-1.0, min(1.0, right_speed))
        
        return left_speed, right_speed

    def _apply_friction_correction(self, speed):
        """Boosts speed slightly if it is below the motor's physical threshold to overcome static friction."""
        if speed == 0.0:
            return 0.0
            
        abs_speed = abs(speed)
        if abs_speed < ControlConfig.MIN_SPEED_THRESHOLD:
            # Scale it up to the minimum threshold preserving direction
            sign = 1.0 if speed > 0.0 else -1.0
            return sign * ControlConfig.MIN_SPEED_THRESHOLD
        return speed

