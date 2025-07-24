from nudenet import NudeDetector
detector = NudeDetector()
# the 320n model included with the package will be used


print(detector.detect('images.jpeg')) # Returns list of detections)
