import time

from picamera2 import Picamera2, Preview

picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})
picam2.configure(preview_config)

picam2.start_preview(Preview.QTGL)

picam2.start()
time.sleep(2)

metadata = picam2.capture_file("test.jpg")
print(metadata)

picam2.close()
