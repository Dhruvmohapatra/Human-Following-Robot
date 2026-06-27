import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np
import psutil
from datetime import datetime

# Import project modules
from config import PinConfig, ControlConfig, CameraConfig, SimConfig, PathConfig
from logger import setup_logger, TelemetryLogger
from motor_controller import MotorController
from ultrasonic import UltrasonicSensor
from camera_tracker import ThreadedCamera, CameraTracker
from follow_controller import FollowController, RoverState

# Set up main system logger
logger = setup_logger("rover", "rover.log", level=logging.INFO)

# Global variables for simulation mouse drag
sim_human_x = SimConfig.INITIAL_HUMAN_X
sim_human_y = SimConfig.INITIAL_HUMAN_Y
dragging_human = False
arena_width = SimConfig.WINDOW_WIDTH // 2
arena_height = SimConfig.WINDOW_HEIGHT

def sim_mouse_callback(event, x, y, flags, param):
    """Callback to allow dragging the virtual human in simulation mode."""
    global sim_human_x, sim_human_y, dragging_human
    # Left click to start drag
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is inside Left Panel (Arena) and near human
        dist = np.sqrt((x - sim_human_x) ** 2 + (y - sim_human_y) ** 2)
        if dist < 20:
            dragging_human = True
            logger.debug("Started dragging virtual human.")
            
    # Mouse move to update coordinates
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_human:
            # Constrain to left panel with small margin
            sim_human_x = max(15, min(arena_width - 15, x))
            sim_human_y = max(15, min(arena_height - 15, y))
            
    # Release left click
    elif event == cv2.EVENT_LBUTTONUP:
        if dragging_human:
            dragging_human = False
            logger.debug("Stopped dragging virtual human.")

def get_ray_intersection(ox, oy, theta, max_range):
    """Calculates simulated distance to arena boundaries (walls) along a ray direction."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    t_min = max_range
    
    # Left wall (x = 0)
    if cos_t < 0:
        t = -ox / cos_t
        if t > 0: t_min = min(t_min, t)
    # Right wall (x = arena_width)
    elif cos_t > 0:
        t = (arena_width - ox) / cos_t
        if t > 0: t_min = min(t_min, t)
        
    # Top wall (y = 0)
    if sin_t < 0:
        t = -oy / sin_t
        if t > 0: t_min = min(t_min, t)
    # Bottom wall (y = arena_height)
    elif sin_t > 0:
        t = (arena_height - oy) / sin_t
        if t > 0: t_min = min(t_min, t)
        
    return t_min

class RoverSystem:
    """Orchestrator for the human following rover system."""
    def __init__(self, sim_mode=False, enable_record=True):
        self.sim_mode = sim_mode
        self.enable_record = enable_record
        self.run_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize Subsystems
        logger.info(f"Initializing Rover Subsystems. Session Name: {self.run_name}")
        self.telemetry_logger = TelemetryLogger(self.run_name)
        
        self.camera = ThreadedCamera(sim_mode=self.sim_mode)
        self.motor = MotorController(sim_mode=self.sim_mode)
        
        self.sensor_front = UltrasonicSensor(
            PinConfig.FRONT_TRIG, PinConfig.FRONT_ECHO, label="Front Ultrasonic", sim_mode=self.sim_mode
        )
        self.sensor_rear = UltrasonicSensor(
            PinConfig.REAR_TRIG, PinConfig.REAR_ECHO, label="Rear Ultrasonic", sim_mode=self.sim_mode
        )
        
        self.tracker = CameraTracker(sim_mode=self.sim_mode)
        self.controller = FollowController()
        
        # Video recording setup
        self.video_writer = None
        self.video_record_path = os.path.join(PathConfig.RUNS_DIR, f"{self.run_name}_cam.avi")
        
        # Simulated states
        self.sim_rover_x = float(SimConfig.INITIAL_ROVER_X)
        self.sim_rover_y = float(SimConfig.INITIAL_ROVER_Y)
        self.sim_rover_theta = float(SimConfig.INITIAL_ROVER_THETA)
        
        # Telemetry metrics
        self.fps = 0.0
        self.last_loop_time = time.time()
        self.running = False
        
        # Setup GUI Window name
        self.window_name = "Human Following Rover Telemetry & Control"

    def start(self):
        """Starts system execution."""
        self.camera.start()
        self.running = True
        logger.info("Rover System successfully started.")
        
        # Create UI Window
        cv2.namedWindow(self.window_name)
        if self.sim_mode:
            cv2.setMouseCallback(self.window_name, sim_mouse_callback)
            
        self._run_loop()

    def _run_loop(self):
        """Core execution loop (runs at ~30Hz target)."""
        target_dt = 1.0 / CameraConfig.FRAME_RATE
        
        while self.running:
            start_tick = time.time()
            
            # Calculate loop dt
            dt = start_tick - self.last_loop_time
            self.last_loop_time = start_tick
            self.fps = 1.0 / dt if dt > 0 else 0.0
            
            # 1. Update Simulation State (if in sim_mode)
            if self.sim_mode:
                self._update_simulation(dt)
                
            # 2. Acquire Sensor Readings
            front_dist = self.sensor_front.read_distance_filtered()
            rear_dist = self.sensor_rear.read_distance_filtered()
            
            # 3. Read camera frame and update tracking
            frame = self.camera.read()
            tracking_success, box = self.tracker.update(frame)
            
            # 4. Run Follow Controller Loop
            left_speed, right_speed = self.controller.update(
                box if tracking_success else None,
                front_dist,
                rear_dist,
                dt
            )
            
            # 5. Apply commands to motors
            self.motor.set_speeds(left_speed, right_speed)
            
            # 6. Session Video Recording & overlays
            display_frame = self._render_ui(
                frame, tracking_success, box, front_dist, rear_dist, left_speed, right_speed
            )
            
            if self.enable_record:
                self._record_video_frame(display_frame)
                
            # 7. Logging Telemetry
            self._log_system_telemetry(front_dist, rear_dist, left_speed, right_speed)
            
            # 8. Check GUI events / Keyboard input
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                logger.info("Shutdown key pressed. Stopping system.")
                self.running = False
                
            # Frame rate pacing
            elapsed = time.time() - start_tick
            sleep_time = max(0.001, target_dt - elapsed)
            time.sleep(sleep_time)
            
        self.shutdown()

    def _update_simulation(self, dt):
        """Simulates differential drive physics and feeds virtual sensor/camera data."""
        global sim_human_x, sim_human_y
        
        # 1. Differential Drive Kinematics
        # Get speeds currently applied to simulated motors
        left_speed, right_speed = self.motor.get_speeds()
        
        # Max velocity conversion (1.5 m/s maximum velocity = 150 pixels/s)
        max_vel = 150.0 # pixels/sec
        v_l = left_speed * max_vel
        v_r = right_speed * max_vel
        
        # Linear and Angular speed
        v = (v_l + v_r) / 2.0
        w = (v_r - v_l) / 35.0  # wheelbase width in pixels
        
        # Update coordinates
        self.sim_rover_x += v * np.cos(self.sim_rover_theta) * dt
        self.sim_rover_y += v * np.sin(self.sim_rover_theta) * dt
        self.sim_rover_theta += w * dt
        
        # Constrain rover to arena boundaries
        self.sim_rover_x = max(20, min(arena_width - 20, self.sim_rover_x))
        self.sim_rover_y = max(20, min(arena_height - 20, self.sim_rover_y))
        
        # 2. Simulated Sensors
        # Front sensor positioned at front edge
        front_offset_x = 18 * np.cos(self.sim_rover_theta)
        front_offset_y = 18 * np.sin(self.sim_rover_theta)
        fx, fy = self.sim_rover_x + front_offset_x, self.sim_rover_y + front_offset_y
        
        # Rear sensor positioned at rear edge
        rx, ry = self.sim_rover_x - front_offset_x, self.sim_rover_y - front_offset_y
        
        # Front Wall intersection
        max_sensor_px = SimConfig.SENSOR_MAX_RANGE_M * SimConfig.PIXELS_PER_METER
        front_wall_dist = get_ray_intersection(fx, fy, self.sim_rover_theta, max_sensor_px)
        # Rear Wall intersection (pointing in theta + pi)
        rear_wall_dist = get_ray_intersection(rx, ry, self.sim_rover_theta + np.pi, max_sensor_px)
        
        # Front distance to human
        dx_f = sim_human_x - fx
        dy_f = sim_human_y - fy
        dist_f = np.sqrt(dx_f**2 + dy_f**2)
        angle_f = np.arctan2(dy_f, dx_f) - self.sim_rover_theta
        # Normalize angle to [-pi, pi]
        angle_f = (angle_f + np.pi) % (2 * np.pi) - np.pi
        
        # Rear distance to human
        dx_r = sim_human_x - rx
        dy_r = sim_human_y - ry
        dist_r = np.sqrt(dx_r**2 + dy_r**2)
        angle_r = np.arctan2(dy_r, dx_r) - (self.sim_rover_theta + np.pi)
        angle_r = (angle_r + np.pi) % (2 * np.pi) - np.pi
        
        # Check FOV (30 degrees)
        fov_rad = np.radians(SimConfig.SENSOR_FOV_DEG)
        
        front_sensor_dist = front_wall_dist
        if abs(angle_f) < fov_rad / 2.0 and dist_f < max_sensor_px:
            front_sensor_dist = min(front_sensor_dist, dist_f)
            
        rear_sensor_dist = rear_wall_dist
        if abs(angle_r) < fov_rad / 2.0 and dist_r < max_sensor_px:
            rear_sensor_dist = min(rear_sensor_dist, dist_r)
            
        # Feed distances (pixels converted to cm: 1 pixel = 1 cm because scale is 100 px/m)
        self.sensor_front.set_simulated_distance(front_sensor_dist)
        self.sensor_rear.set_simulated_distance(rear_sensor_dist)
        
        # 3. Simulated Camera Frame & Tracker Box
        # Camera FOV horizontal: 60 degrees (1.05 radians)
        cam_fov = 1.05
        dx_cam = sim_human_x - self.sim_rover_x
        dy_cam = sim_human_y - self.sim_rover_y
        dist_cam = np.sqrt(dx_cam**2 + dy_cam**2)
        angle_cam = np.arctan2(dy_cam, dx_cam) - self.sim_rover_theta
        angle_cam = (angle_cam + np.pi) % (2 * np.pi) - np.pi
        
        # Generate simulated camera canvas
        cam_w, cam_h = CameraConfig.FRAME_WIDTH, CameraConfig.FRAME_HEIGHT
        cam_frame = np.ones((cam_h, cam_w, 3), dtype=np.uint8) * 40 # Dark Gray background
        
        box = None
        # If human in front camera FOV
        if abs(angle_cam) < cam_fov / 2.0 and dist_cam < 600:
            # Horizontal index
            cx = (cam_w / 2.0) + (cam_w / 2.0) * (angle_cam / (cam_fov / 2.0))
            # Height of bounding box inversely proportional to distance (meters)
            dist_meters = dist_cam / 100.0
            
            # Calibration: human is ~1.7m tall.
            # Bounding box height ratio = Desired ratio (0.5) when human is at 3.0 meters.
            # Thus, ratio = 1.5 / dist_meters
            height_ratio = 1.5 / max(0.5, dist_meters)
            box_h = int(cam_h * height_ratio)
            box_w = int(box_h * 0.5) # aspect ratio
            
            box_x = int(cx - box_w / 2)
            box_y = int(cam_h / 2 - box_h / 2)
            box = (box_x, box_y, box_w, box_h)
            
            # Draw a simulated human stick-figure/box representation on camera frame
            cv2.rectangle(cam_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
            cv2.circle(cam_frame, (int(cx), box_y + int(box_h * 0.15)), int(box_w * 0.25), (0, 200, 0), -1) # Head
            cv2.line(cam_frame, (int(cx), box_y + int(box_h * 0.3)), (int(cx), box_y + int(box_h * 0.75)), (0, 200, 0), 3) # Body
            cv2.line(cam_frame, (int(cx) - int(box_w * 0.35), box_y + int(box_h * 0.45)), (int(cx) + int(box_w * 0.35), box_y + int(box_h * 0.45)), (0, 200, 0), 2) # Arms
            cv2.line(cam_frame, (int(cx), box_y + int(box_h * 0.75)), (box_x, box_y + box_h), (0, 200, 0), 2) # Left leg
            cv2.line(cam_frame, (int(cx), box_y + int(box_h * 0.75)), (box_x + box_w, box_y + box_h), (0, 200, 0), 2) # Right leg
            
        self.camera.set_simulated_frame(cam_frame)
        self.tracker.tracking_box = box

    def _render_ui(self, frame, tracking_success, box, front_dist, rear_dist, left_speed, right_speed):
        """Assembles the visual display showing camera feed, telemetry dashboard, and simulation arena (if enabled)."""
        # Dimensions
        arena_w, arena_h = arena_width, arena_height
        panel_w = SimConfig.WINDOW_WIDTH - arena_w
        
        # 1. Render Left Panel (Arena) in simulation mode
        if self.sim_mode:
            arena_canvas = np.ones((arena_h, arena_w, 3), dtype=np.uint8) * 30  # Very dark slate gray
            
            # Draw walls grid
            for i in range(0, arena_w, 40):
                cv2.line(arena_canvas, (i, 0), (i, arena_h), (40, 40, 40), 1)
            for j in range(0, arena_h, 40):
                cv2.line(arena_canvas, (0, j), (arena_w, j), (40, 40, 40), 1)
                
            # Draw Sensor FOV cones
            fov_deg = SimConfig.SENSOR_FOV_DEG
            r_x, r_y = int(self.sim_rover_x), int(self.sim_rover_y)
            r_theta = self.sim_rover_theta
            
            # Front sensor cone
            f_cone_left = r_theta - np.radians(fov_deg/2)
            f_cone_right = r_theta + np.radians(fov_deg/2)
            fx = int(r_x + 18 * np.cos(r_theta))
            fy = int(r_y + 18 * np.sin(r_theta))
            f_pt_l = (int(fx + 100 * np.cos(f_cone_left)), int(fy + 100 * np.sin(f_cone_left)))
            f_pt_r = (int(fx + 100 * np.cos(f_cone_right)), int(fy + 100 * np.sin(f_cone_right)))
            cv2.line(arena_canvas, (fx, fy), f_pt_l, (100, 150, 100), 1, cv2.LINE_AA)
            cv2.line(arena_canvas, (fx, fy), f_pt_r, (100, 150, 100), 1, cv2.LINE_AA)
            
            # Rear sensor cone
            r_cone_left = r_theta + np.pi - np.radians(fov_deg/2)
            r_cone_right = r_theta + np.pi + np.radians(fov_deg/2)
            rx = int(r_x - 18 * np.cos(r_theta))
            ry = int(r_y - 18 * np.sin(r_theta))
            r_pt_l = (int(rx + 100 * np.cos(r_cone_left)), int(ry + 100 * np.sin(r_cone_left)))
            r_pt_r = (int(rx + 100 * np.cos(r_cone_right)), int(ry + 100 * np.sin(r_cone_right)))
            cv2.line(arena_canvas, (rx, ry), r_pt_l, (100, 100, 150), 1, cv2.LINE_AA)
            cv2.line(arena_canvas, (rx, ry), r_pt_r, (100, 100, 150), 1, cv2.LINE_AA)
            
            # Draw Rover as rotated rectangle
            rect = ((r_x, r_y), (35, 25), np.degrees(r_theta))
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            cv2.drawContours(arena_canvas, [box_points], 0, (220, 150, 50), -1)  # Rover body
            cv2.drawContours(arena_canvas, [box_points], 0, (255, 255, 255), 2)  # Rover outline
            
            # Draw wheels (4 small rectangles)
            for pt in box_points:
                cv2.circle(arena_canvas, tuple(pt), 4, (0, 0, 0), -1)
                
            # Rover heading indicator line
            hx, hy = int(r_x + 25 * np.cos(r_theta)), int(r_y + 25 * np.sin(r_theta))
            cv2.line(arena_canvas, (r_x, r_y), (hx, hy), (0, 0, 255), 2, cv2.LINE_AA)
            
            # Draw Target Human
            h_x, h_y = int(sim_human_x), int(sim_human_y)
            cv2.circle(arena_canvas, (h_x, h_y), 12, (0, 255, 0), -1)  # Inner circle
            cv2.circle(arena_canvas, (h_x, h_y), 18, (0, 255, 0), 2)  # Target ring
            cv2.putText(arena_canvas, "DRAG ME", (h_x - 30, h_y - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Section header
            cv2.putText(arena_canvas, "2D TOP-DOWN ARENA SIMULATOR", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            # On physical mode, left panel is not used, we just show a placeholder or adjust dimensions
            # Let's create a black canvas
            arena_canvas = np.zeros((arena_h, arena_w, 3), dtype=np.uint8)
            cv2.putText(arena_canvas, "PHYSICAL DEPLOYMENT ACTIVE", (20, arena_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # 2. Render Right Panel (Camera feed & Telemetry dashboard)
        # Resize camera frame to fit top portion of the right panel
        cam_panel_h = 384
        cam_panel_w = panel_w
        cam_resized = cv2.resize(frame, (cam_panel_w, cam_panel_h))
        
        # Bounding box overlays on camera panel
        if tracking_success and box:
            # Since frame was resized, scale the bounding box
            scale_x = cam_panel_w / float(CameraConfig.FRAME_WIDTH)
            scale_y = cam_panel_h / float(CameraConfig.FRAME_HEIGHT)
            bx, by, bw, bh = box
            bx, by = int(bx * scale_x), int(by * scale_y)
            bw, bh = int(bw * scale_x), int(bh * scale_y)
            
            cv2.rectangle(cam_resized, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(cam_resized, "TARGET LOCK", (bx, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            
        # Draw Camera panel label
        cv2.putText(cam_resized, "VIRTUAL CAMERA" if self.sim_mode else "LIVE CAMERA", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Telemetry panel (bottom half of right panel)
        telem_h = arena_height - cam_panel_h
        telem_canvas = np.ones((telem_h, panel_w, 3), dtype=np.uint8) * 15 # Dark background
        
        # Telemetry background accent line
        cv2.line(telem_canvas, (0, 0), (panel_w, 0), (50, 50, 50), 2)
        
        # Get metrics
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        det_lat = self.tracker.detection_latency
        trk_lat = self.tracker.tracking_latency
        
        # State Colors
        state_colors = {
            RoverState.INITIALIZING: (200, 200, 200),
            RoverState.SEARCHING: (0, 165, 255),   # Orange
            RoverState.TRACKING: (255, 255, 0),    # Cyan
            RoverState.FOLLOWING: (0, 255, 0),     # Green
            RoverState.BLOCKED: (0, 0, 255),       # Red
            RoverState.LOST_TARGET: (0, 255, 255), # Yellow
            RoverState.SHUTDOWN: (128, 128, 128)
        }
        state_color = state_colors.get(self.controller.state, (255, 255, 255))
        
        # Draw Text Layout
        y_offset = 30
        line_spacing = 26
        
        def put_metric(label, value, val_color=(255, 255, 255)):
            nonlocal y_offset
            cv2.putText(telem_canvas, f"{label}:", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(telem_canvas, str(value), (180, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, val_color, 1, cv2.LINE_AA)
            y_offset += line_spacing

        put_metric("ROVER STATE", self.controller.state, val_color=state_color)
        put_metric("LOOP FREQUENCY", f"{self.fps:.1f} FPS")
        put_metric("DETECTION LATENCY", f"{det_lat:.1f} ms" if not self.sim_mode else "MOCK")
        put_metric("TRACKING LATENCY", f"{trk_lat:.1f} ms" if not self.sim_mode else "MOCK")
        put_metric("FRONT SENSOR DIST", f"{front_dist:.1f} cm", (0, 255, 0) if front_dist > 25 else (0, 0, 255))
        put_metric("REAR SENSOR DIST", f"{rear_dist:.1f} cm", (0, 255, 0) if rear_dist > 25 else (0, 0, 255))
        put_metric("MOTOR LEFT SPEED", f"{left_speed:.2f}", (0, 255, 255) if left_speed != 0.0 else (128, 128, 128))
        put_metric("MOTOR RIGHT SPEED", f"{right_speed:.2f}", (0, 255, 255) if right_speed != 0.0 else (128, 128, 128))
        put_metric("CPU UTILIZATION", f"{cpu_usage:.1f}%")
        put_metric("MEMORY UTILIZATION", f"{mem_usage:.1f}%")
        
        # Build composite output window
        right_panel = np.vstack([cam_resized, telem_canvas])
        composite_window = np.hstack([arena_canvas, right_panel])
        
        return composite_window

    def _record_video_frame(self, frame):
        """Writes current UI composite frame to video writer file."""
        h, w = frame.shape[:2]
        if self.video_writer is None:
            # Define codec and initialize VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            logger.info(f"Starting session recording at: {self.video_record_path}")
            self.video_writer = cv2.VideoWriter(self.video_record_path, fourcc, 15.0, (w, h))
            
        try:
            self.video_writer.write(frame)
        except Exception as e:
            # Prevent log spam on recording issues
            pass

    def _log_system_telemetry(self, front_dist, rear_dist, left_speed, right_speed):
        """Logs live numeric parameters to the TelemetryLogger (CSV)."""
        det_lat = self.tracker.detection_latency
        trk_lat = self.tracker.tracking_latency
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        
        telemetry_payload = {
            "state": self.controller.state,
            "fps": round(self.fps, 2),
            "detection_latency_ms": round(det_lat, 2) if not self.sim_mode else 0.0,
            "tracking_latency_ms": round(trk_lat, 2) if not self.sim_mode else 0.0,
            "cpu_usage_pct": cpu_usage,
            "memory_usage_pct": mem_usage,
            "front_distance_cm": round(front_dist, 2),
            "rear_distance_cm": round(rear_dist, 2),
            "motor_left_speed": round(left_speed, 2),
            "motor_right_speed": round(right_speed, 2)
        }
        
        self.telemetry_logger.log(telemetry_payload)

    def shutdown(self):
        """Performs ordered shutdown of all modules."""
        logger.info("Executing System Shutdown...")
        
        # Update FSM state
        self.controller.change_state(RoverState.SHUTDOWN)
        
        # Close Video Writer
        if self.video_writer:
            try:
                self.video_writer.release()
                logger.info(f"Video session recording saved to {self.video_record_path}")
            except Exception:
                pass
            self.video_writer = None
            
        # Close CSV logger
        self.telemetry_logger.close()
        
        # Release Subsystems
        self.camera.release()
        self.motor.cleanup()
        self.sensor_front.close()
        self.sensor_rear.close()
        
        # Destroy GUI
        cv2.destroyAllWindows()
        logger.info("System Shutdown Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Following Rover Vision & Control System")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Force simulation mode (runs without physical RPi hardware)"
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable session video recording"
    )
    args = parser.parse_args()
    
    # Check platform to auto-toggle simulation mode
    is_rpi = False
    try:
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                if "raspberry pi" in model:
                    is_rpi = True
    except Exception:
        pass

    # If not on Raspberry Pi, force simulation mode
    sim_flag = args.sim or (not is_rpi)
    if not is_rpi and not args.sim:
        logger.info("Non-RPi hardware detected. Auto-toggling simulation mode.")

    rover = RoverSystem(sim_mode=sim_flag, enable_record=not args.no_record)
    
    try:
        rover.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        rover.shutdown()
    except Exception as e:
        logger.critical(f"Unhandled system crash: {e}", exc_info=True)
        rover.shutdown()
        sys.exit(1)

