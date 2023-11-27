from dataclasses import dataclass

import cv2

from leap import HandType
from leap.cstruct import LeapCStruct
from leap.datatypes import Digit, Hand
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


def extract_hand_data(hand: Hand):
    """
    Extract data from a hand. This includes the hand data, palm data and each digit data.
    :param hand: hand to extract data from
    :return: list of extracted data
    """
    hand_data = extract_data(hand, hand_data_headers)
    palm_data = extract_data(hand.palm, palm_data_headers)
    hand_data.extend(palm_data)
    for digit in hand.digits:
        digit_data = extract_digit_data(digit)
        hand_data.extend(digit_data)
    return hand_data


@dataclass
class LeapCStructMock:
    """
    Mock class to replace LeapCStruct. Not guaranteed to have all the same fields.
    The _data_headers lists should inform what fields are expected.
    """
    pass


def roll_up_data(extracted_data_reversed: list, headers: list[str]):
    """
    Roll up data from a list of headers into a LeapCStructMock.
    :param extracted_data_reversed: Reverse list of data to consume
    :param headers: List of headers to consume
    :return: LeapCStructMock containing the rolled up data
    """
    struct = LeapCStructMock()
    for header in headers:
        root = struct
        while "." in header:
            root_header, sub_header = header.split(".", maxsplit=1)
            if not hasattr(root, root_header):
                setattr(root, root_header, LeapCStructMock())
            root = getattr(root, root_header)
            header = sub_header
        setattr(root, header, extracted_data_reversed.pop())
    return struct


def roll_up_digit_data(extracted_data_reversed: list):
    """
    Roll up digit data into a Digit.
    :param extracted_data_reversed: Reverse list of data to consume
    :return: Digit containing the rolled up data
    """
    digit_data = roll_up_data(extracted_data_reversed, digit_data_headers)
    # the ordering of bones matters and MUST match the Digit.bones ordering
    for bone in ["metacarpal", "proximal", "intermediate", "distal"]:
        bone_data = roll_up_data(extracted_data_reversed, bone_data_headers)
        setattr(digit_data, bone, bone_data)
    return digit_data


def roll_up_hand_data(extracted_hand_data: list):
    """
    Roll up extracted hand data into a complete Hand object, including Hand, Palm, Digit and Bone data.
    :param extracted_hand_data: List of extracted hand data to consume
    :return: Hand containing the rolled up data
    """
    extracted_data_reversed = extracted_hand_data[::-1]

    # Roll up hand data
    hand_data = roll_up_data(extracted_data_reversed, hand_data_headers)

    # Roll up palm data
    palm_data = roll_up_data(extracted_data_reversed, palm_data_headers)
    setattr(hand_data, "palm", palm_data)

    # Roll up digit data
    # the ordering of digits matters and MUST match the Hand.digits ordering
    for digit in ["thumb", "index", "middle", "ring", "pinky"]:
        digit_data = roll_up_digit_data(extracted_data_reversed)
        setattr(hand_data, digit, digit_data)

    # should've consumed all the data
    assert len(extracted_data_reversed) == 0

    return Hand(hand_data)


def main():
    try:
        with LeapMotionTracker() as tracker:
            while True:
                if tracker.has_new_event:
                    tracker.render_hands(tracker.event.hands)
                    cv2.imshow(tracker.name, tracker.output_image)
                    key = cv2.waitKey(1)

                    if key == ord("x"):
                        break
                    elif key == ord("f"):
                        if tracker.hands_format == "Skeleton":
                            tracker.hands_format = "Dots"
                        else:
                            tracker.hands_format = "Skeleton"

                    for hand in tracker.event.hands:
                        if hand.type == HandType.Left:
                            continue  # ignore left hand for now
                        # get the data from the tracker, flatten it to extract the hand data
                        hand_data = extract_hand_data(hand)
                        # take the flattened/extracted data and roll it back up into a Hand object
                        hand = roll_up_hand_data(hand_data)
                        # take the rolled up Hand object and flatten it again
                        hand_data_test = extract_hand_data(hand)
                        # we should have the same data as before (that we care about)
                        assert hand_data == hand_data_test


    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
