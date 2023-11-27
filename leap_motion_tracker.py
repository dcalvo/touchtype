import time

import cv2
import numpy as np

import leap
from leap.datatypes import Hand
from leap.events import TrackingEvent
from visualizer import Visualizer

_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}


class LeapMotionTracker(leap.Listener):
    def __init__(self):
        self.connection = leap.Connection()
        self.connection.add_listener(self)
        self.has_new_event = False
        self._event = None

    def __enter__(self):
        self.connection.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.disconnect()

    def on_connection_event(self, event):
        pass

    def on_tracking_mode_event(self, event):
        print(f"Tracking mode changed to {_TRACKING_MODES[event.current_tracking_mode]}")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        self._event = event
        self.has_new_event = True

    @property
    def event(self) -> TrackingEvent:
        if self._event is None:
            raise ValueError("No event has been received yet")
        self.has_new_event = False
        return self._event


def main():
    viz = Visualizer()
    with LeapMotionTracker() as tracker:
        while True:
            time.sleep(1)  # test sleep delay when polling LeapMotionTracker
            if tracker.has_new_event:
                viz.render_hands(tracker.event.hands)
                if viz.show():
                    break


if __name__ == "__main__":
    main()
