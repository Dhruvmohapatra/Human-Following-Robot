import os
import sys
import logging
import csv
from datetime import datetime
from config import PathConfig

def setup_logger(name="rover", log_file="rover.log", level=logging.INFO):
    """Sets up a logger with handlers for both console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    # Formatters
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'
    )
    console_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # File Handler
    log_path = os.path.join(PathConfig.LOGS_DIR, log_file)
    try:
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Failed to create file log handler: {e}", file=sys.stderr)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

class TelemetryLogger:
    """Logs structured telemetry data to CSV for session analysis."""
    def __init__(self, run_name=None):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.filename = os.path.join(PathConfig.RUNS_DIR, f"{run_name}_telemetry.csv")
        self.csv_file = None
        self.csv_writer = None
        
        # Define fields to match requested telemetry data
        self.fieldnames = [
            "timestamp",
            "state",
            "fps",
            "detection_latency_ms",
            "tracking_latency_ms",
            "cpu_usage_pct",
            "memory_usage_pct",
            "front_distance_cm",
            "rear_distance_cm",
            "motor_left_speed",
            "motor_right_speed"
        ]
        
        self._initialize_csv()

    def _initialize_csv(self):
        try:
            self.csv_file = open(self.filename, mode='w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()
            self.csv_file.flush()
        except Exception as e:
            print(f"Error initializing telemetry CSV file: {e}", file=sys.stderr)

    def log(self, telemetry_data):
        """
        Logs a row of telemetry data.
        telemetry_data: dict containing keys corresponding to fieldnames (excluding timestamp)
        """
        if not self.csv_file or not self.csv_writer:
            return
            
        row = {field: "" for field in self.fieldnames}
        row["timestamp"] = datetime.now().isoformat()
        
        for field in self.fieldnames:
            if field in telemetry_data:
                row[field] = telemetry_data[field]
                
        try:
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        except Exception as e:
            # Silent fail during high-frequency loop to avoid crashing main execution
            pass

    def close(self):
        """Safely closes the telemetry file."""
        if self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None
            self.csv_writer = None

