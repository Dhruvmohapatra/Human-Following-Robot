import os
import time
import logging
import threading
import requests
import cv2
import numpy as np
from config import CameraConfig, PathConfig

logger = logging.getLogger("rover")

def download_file(url, dest_path):
    """Downloads a file from a URL to a local destination path."""
    logger.info(f"Downloading {os.path.basename(dest_path)} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Successfully downloaded {os.path.basename(dest_path)}.")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def get_opencv_tracker(tracker_type):
    """Factory to create an OpenCV tracker, compatible across various OpenCV versions."""
    tracker_type = tracker_type.upper()
    
    # Modern OpenCV 4.5.1+ tracker creation syntax
    if hasattr(cv2, "TrackerKCF_create") and tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    elif hasattr(cv2, "TrackerCSRT_create") and tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, "TrackerMIL_create") and tracker_type == "MIL":
        return cv2.TrackerMIL_create()
        
    # Legacy OpenCV syntax fallback
    if hasattr(cv2, "legacy"):
        try:
            if tracker_type == "KCF":
                return cv2.legacy.TrackerKCF_create()
            elif tracker_type == "CSRT":
                return cv2.legacy.TrackerCSRT_create()
            elif tracker_type == "MIL":
                return cv2.legacy.TrackerMIL_create()
        except AttributeError:
            pass
            
    # Default fallback
    try:
        return cv2.TrackerKCF_create()
    except AttributeError:
        return cv2.legacy.TrackerKCF_create()


class ThreadedCamera:
    """Threaded camera wrapper to read frames asynchronously and reduce latency."""
    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.width = CameraConfig.FRAME_WIDTH
        self.height = CameraConfig.FRAME_HEIGHT
        self.fps = CameraConfig.FRAME_RATE
        
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Hardware sources
        self.cap = None
        self.picam = None
        
        if not self.sim_mode:
            self._init_hardware_camera()

    def _init_hardware_camera(self):
        # First, try to import and initialize Picamera2 (Raspberry Pi OS Bookworm)
        try:
            from picamera2 import Picamera2
            logger.info("Initializing Picamera2...")
            self.picam = Picamera2()
            
            # Configure camera
            config = self.picam.create_preview_configuration(main={"size": (self.width, self.height)})
            self.picam.configure(config)
            self.picam.start()
            logger.info("Picamera2 initialized successfully.")
            return
        except (ImportError, Exception) as e:
            logger.info(f"Picamera2 not available ({e}). Trying OpenCV VideoCapture...")
            
        # Fallback: OpenCV VideoCapture (e.g., USB camera / WebCam)
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                logger.info("OpenCV VideoCapture initialized successfully.")
                return
            else:
                if self.cap:
                    self.cap.release()
                raise RuntimeError("Failed to read test frame from VideoCapture.")
        except Exception as ex:
            logger.warning(f"No hardware camera detected ({ex}). Running in simulated camera mode.")
            self.sim_mode = True

    def start(self):
        """Starts the background frame capture thread."""
        if self.running:
            return
            
        self.running = True
        if self.sim_mode:
            # Simulated camera uses canvas updates from main loop
            self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            logger.info("Simulated camera thread active.")
            return
            
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.name = "CameraCaptureThread"
        self.thread.start()
        logger.info("Camera capture thread started.")

    def _capture_loop(self):
        while self.running:
            frame_raw = None
            if self.picam:
                try:
                    frame_raw = self.picam.capture_array()
                    # Picamera2 array is often RGB, OpenCV defaults to BGR
                    if frame_raw is not None:
                        frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.error(f"Picamera2 capture error: {e}")
            elif self.cap:
                try:
                    ret, frame_raw = self.cap.read()
                    if not ret:
                        frame_raw = None
                except Exception as e:
                    logger.error(f"OpenCV VideoCapture error: {e}")
                    
            if frame_raw is not None:
                with self.lock:
                    self.frame = frame_raw.copy()
                    
            # Avoid burning CPU
            time.sleep(1.0 / self.fps)

    def set_simulated_frame(self, frame):
        """Injects a frame when running in simulation mode."""
        if self.sim_mode:
            with self.lock:
                self.frame = frame.copy()

    def read(self):
        """Returns the latest frame."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self):
        """Stops the camera thread and releases hardware handles."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        if self.picam:
            try:
                self.picam.stop()
                self.picam.close()
            except Exception:
                pass
            self.picam = None
            
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        logger.info("Camera resources released.")


class CameraTracker:
    """Manages the vision processing: MobileNet SSD detection and OpenCV Tracking."""
    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.tracker_type = CameraConfig.TRACKER_TYPE
        
        # State tracking
        self.tracker = None
        self.tracking_box = None  # (x, y, w, h)
        self.is_tracking = False
        self.frame_counter = 0
        
        # Performance metrics
        self.detection_latency = 0.0
        self.tracking_latency = 0.0
        
        # MobileNet SSD setup
        self.net = None
        if not self.sim_mode:
            self._initialize_detector()

    def _initialize_detector(self):
        # Check if files exist, download if necessary
        txt_exists = os.path.exists(PathConfig.MODEL_TXT)
        weights_exists = os.path.exists(PathConfig.MODEL_WEIGHTS)
        
        if not txt_exists:
            download_file(PathConfig.MODEL_TXT_URL, PathConfig.MODEL_TXT)
        if not weights_exists:
            download_file(PathConfig.MODEL_WEIGHTS_URL, PathConfig.MODEL_WEIGHTS)
            
        try:
            logger.info("Loading MobileNet SSD DNN model...")
            self.net = cv2.dnn.readNetFromCaffe(PathConfig.MODEL_TXT, PathConfig.MODEL_WEIGHTS)
            # Set preferable backend to opencl or default CPU target
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("MobileNet SSD loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Caffe model: {e}. Falling back to simulation detection logic.")
            self.sim_mode = True

    def detect_person(self, frame):
        """
        Runs MobileNet SSD inference to find a person.
        Returns bounding box (x, y, w, h) or None if no person is found.
        """
        if self.sim_mode:
            # In simulation, detection coordinates are set externally by the simulation loop
            return self.tracking_box

        start_time = time.time()
        h, w = frame.shape[:2]
        
        # Prepare 300x300 image blob for MobileNet SSD
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        best_box = None
        max_confidence = 0.0
        
        # Loop through detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CameraConfig.MODEL_CONFIDENCE_THRESHOLD:
                class_id = int(detections[0, 0, i, 1])
                if class_id == CameraConfig.MODEL_CLASS_PERSON_ID:
                    # Scale coordinates to frame size
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Convert to (x, y, w, h)
                    box_w = endX - startX
                    box_h = endY - startY
                    
                    # Target center calculation relative to frame center
                    # We want to pick the person with highest confidence
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_box = (startX, startY, box_w, box_h)
                        
        self.detection_latency = (time.time() - start_time) * 1000.0 # ms
        
        if best_box:
            logger.info(f"Person detected! Confidence: {max_confidence:.2f}, Box: {best_box}")
            return best_box
        return None

    def initialize_tracker(self, frame, box):
        """Initializes OpenCV tracker on the specified box."""
        if self.sim_mode:
            self.tracking_box = box
            self.is_tracking = True
            return True

        self.tracker = get_opencv_tracker(self.tracker_type)
        try:
            # OpenCV tracker requires (x, y, w, h) as tuple
            success = self.tracker.init(frame, tuple(box))
            if success:
                self.tracking_box = box
                self.is_tracking = True
                self.frame_counter = 0
                logger.info(f"OpenCV {self.tracker_type} Tracker initialized on {box}")
                return True
        except Exception as e:
            logger.error(f"Error initializing tracker: {e}")
            
        self.is_tracking = False
        return False

    def update(self, frame):
        """
        Updates tracking.
        Runs tracker update, or redundant detection every N frames to avoid drift.
        Returns: (is_tracking_successful, bounding_box_tuple)
        """
        if not self.is_tracking:
            # Not tracking, need to detect first
            box = self.detect_person(frame)
            if box:
                success = self.initialize_tracker(frame, box)
                return success, self.tracking_box
            return False, None

        # Redundant detection check to prevent drift (only in physical hardware mode)
        if not self.sim_mode:
            self.frame_counter += 1
            if self.frame_counter >= CameraConfig.REDETECTION_INTERVAL_FRAMES:
                self.frame_counter = 0
                logger.debug("Running periodic detection to correct tracker drift...")
                box = self.detect_person(frame)
                if box:
                    # Re-initialize tracker with the newly detected correct box
                    self.initialize_tracker(frame, box)
                    return True, self.tracking_box

        # Standard Tracker Update
        if self.sim_mode:
            # Bounding box is updated by simulation environment
            return True, self.tracking_box

        start_time = time.time()
        try:
            success, box = self.tracker.update(frame)
            self.tracking_latency = (time.time() - start_time) * 1000.0 # ms
            
            if success:
                # box is returned as floats: (x, y, w, h)
                self.tracking_box = tuple(map(int, box))
                return True, self.tracking_box
            else:
                logger.warning("Tracker lost target!")
                self.is_tracking = False
                self.tracking_box = None
                return False, None
        except Exception as e:
            logger.error(f"Tracker update exception: {e}")
            self.is_tracking = False
            self.tracking_box = None
            return False, None

    def reset(self):
        """Resets the tracking state."""
        self.tracker = None
        self.tracking_box = None
        self.is_tracking = False
        self.frame_counter = 0

