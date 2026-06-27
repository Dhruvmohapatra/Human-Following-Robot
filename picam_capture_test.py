from picamera2 import Picamera2
import time

cam = Picamera2()
cfg = cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
cam.configure(cfg)
cam.start()
time.sleep(1)
frame = cam.capture_array()
print("CAPTURE_OK", frame.shape, frame.dtype)
cam.stop()
cam.close()
