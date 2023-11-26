import cv2
import torch
import numpy as np

from leap import HandType
from leap.cstruct import LeapCStruct
from leap.datatypes import Digit, Bone, Palm, Hand, Vector
from leap.events import TrackingEvent
from leap_motion_tracker import LeapMotionTracker

hand_data_headers = [
    # "id",
    # "flags",
    # "type",
    "confidence",
    # "visible_time",
    "pinch_distance",
    "grab_angle",
    "pinch_strength",
    "grab_strength",
]

palm_data_headers = [
    "position.x",
    "position.y",
    "position.z",
    "stabilized_position.x",
    "stabilized_position.y",
    "stabilized_position.z",
    "velocity.x",
    "velocity.y",
    "velocity.z",
    "normal.x",
    "normal.y",
    "normal.z",
    "width",
    "direction.x",
    "direction.y",
    "direction.z",
    "orientation.x",
    "orientation.y",
    "orientation.z",
    "orientation.w",
]

digit_data_headers = [
    # "finger_id",
    "is_extended",
]

bone_data_headers = [
    "prev_joint.x",
    "prev_joint.y",
    "prev_joint.z",
    "next_joint.x",
    "next_joint.y",
    "next_joint.z",
    "width",
    "rotation.x",
    "rotation.y",
    "rotation.z",
    "rotation.w",
]


def extract_data(struct: LeapCStruct, headers: list[str]):
    """
    Extract data from a LeapCStruct using a list of headers.
    If the struct is nested, the headers can be nested using dot notation.
    e.g. "position.x" will extract the x value from the position vector.
    :param struct: struct to extract data from
    :param headers: list of headers to extract
    :return: list of extracted data
    """
    data = []
    for header in headers:
        root = struct
        while "." in header:
            root_header, sub_header = header.split(".", maxsplit=1)
            root = getattr(root, root_header)
            header = sub_header
        data.append(getattr(root, header))
    return data


def extract_digit_data(digit: Digit):
    """
    Extract data from a digit. This includes the digit data and each bone data.
    :param digit: digit to extract data from
    :return: list of extracted data
    """
    digit_data = extract_data(digit, digit_data_headers)
    for bone in digit.bones:
        bone_data = extract_data(bone, bone_data_headers)
        digit_data.extend(bone_data)
    return digit_data


def extract_event_data(event: TrackingEvent):
    """
    Extract data from a TrackingEvent. Flatten the data from each hand, palm, digit and bone.
    :param event: event to extract data from
    :return: list of extracted data
    """
    for hand in event.hands:
        # Ignore left hand (for now)
        if hand.type == HandType.Left:
            continue

        # Extract hand data
        hand_data = extract_data(hand, hand_data_headers)

        # Extract palm data
        palm_data = extract_data(hand.palm, palm_data_headers)

        # Extract thumb data
        thumb_data = extract_digit_data(hand.thumb)

        # Extract index data
        index_data = extract_digit_data(hand.index)

        # Extract middle data
        middle_data = extract_digit_data(hand.middle)

        # Extract ring data
        ring_data = extract_digit_data(hand.ring)

        # Extract pinky data
        pinky_data = extract_digit_data(hand.pinky)

        # Flatten data
        data = [
            *hand_data,
            *palm_data,
            *thumb_data,
            *index_data,
            *middle_data,
            *ring_data,
            *pinky_data,
        ]

        return data


def roll_up_event_data(event_data: list):
    # Roll up hand data
    pass


def main():
    tracker = LeapMotionTracker()
    tracker.run()
    try:
        while True:
            if tracker.most_recent_event:
                tracker.render_hands(tracker.most_recent_event)
                cv2.imshow(tracker.name, tracker.output_image)
                key = cv2.waitKey(1)

                if key == ord("x"):
                    break
                elif key == ord("f"):
                    if tracker.hands_format == "Skeleton":
                        tracker.hands_format = "Dots"
                    else:
                        tracker.hands_format = "Skeleton"

                data = extract_event_data(tracker.most_recent_event)
                if data:
                    print(len(data))
                    print(data)


    except Exception as e:
        raise e
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
