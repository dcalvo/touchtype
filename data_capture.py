import cv2
import torch
import numpy as np

from leap import HandType
from leap.datatypes import Digit, Bone, Palm, Hand
from leap.events import TrackingEvent
from leap_motion_tracker import LeapMotionTracker


def extract_bone_data(bone: Bone):
    # Extract bone data
    bone_prev_joint_x = bone.prev_joint.x
    bone_prev_joint_y = bone.prev_joint.y
    bone_prev_joint_z = bone.prev_joint.z
    bone_next_joint_x = bone.next_joint.x
    bone_next_joint_y = bone.next_joint.y
    bone_next_joint_z = bone.next_joint.z
    bone_width = bone.width
    bone_rotation_x = bone.rotation.x
    bone_rotation_y = bone.rotation.y
    bone_rotation_z = bone.rotation.z
    bone_rotation_w = bone.rotation.w

    data = [
        bone_prev_joint_x,
        bone_prev_joint_y,
        bone_prev_joint_z,
        bone_next_joint_x,
        bone_next_joint_y,
        bone_next_joint_z,
        bone_width,
        bone_rotation_x,
        bone_rotation_y,
        bone_rotation_z,
        bone_rotation_w,
    ]

    return data


def extract_digit_data(digit: Digit):
    # Extract digit data
    digit_id = digit.finger_id
    digit_is_extended = digit.is_extended

    # Extract metacarpal data
    metacarpal_data = extract_bone_data(digit.metacarpal)

    # Extract proximal data
    proximal_data = extract_bone_data(digit.proximal)

    # Extract intermediate data
    intermediate_data = extract_bone_data(digit.intermediate)

    # Extract distal data
    distal_data = extract_bone_data(digit.distal)

    data = [
        # digit_id,
        digit_is_extended,
        *metacarpal_data,
        *proximal_data,
        *intermediate_data,
        *distal_data,
    ]

    return data


def extract_palm_data(palm: Palm):
    # Extract palm data
    palm_position_x = palm.position.x
    palm_position_y = palm.position.y
    palm_position_z = palm.position.z
    palm_stabilized_position_x = palm.stabilized_position.x
    palm_stabilized_position_y = palm.stabilized_position.y
    palm_stabilized_position_z = palm.stabilized_position.z
    palm_velocity_x = palm.velocity.x
    palm_velocity_y = palm.velocity.y
    palm_velocity_z = palm.velocity.z
    palm_normal_x = palm.normal.x
    palm_normal_y = palm.normal.y
    palm_normal_z = palm.normal.z
    palm_width = palm.width
    palm_direction_x = palm.direction.x
    palm_direction_y = palm.direction.y
    palm_direction_z = palm.direction.z
    palm_orientation_x = palm.orientation.x
    palm_orientation_y = palm.orientation.y
    palm_orientation_z = palm.orientation.z
    palm_orientation_w = palm.orientation.w

    data = [
        palm_position_x,
        palm_position_y,
        palm_position_z,
        palm_stabilized_position_x,
        palm_stabilized_position_y,
        palm_stabilized_position_z,
        palm_velocity_x,
        palm_velocity_y,
        palm_velocity_z,
        palm_normal_x,
        palm_normal_y,
        palm_normal_z,
        palm_width,
        palm_direction_x,
        palm_direction_y,
        palm_direction_z,
        palm_orientation_x,
        palm_orientation_y,
        palm_orientation_z,
        palm_orientation_w,
    ]

    return data


def extract_hand_data(hand: Hand):
    # Extract hand data
    hand_id = hand.id
    hand_flags = hand.flags
    hand_type = hand.type
    hand_confidence = hand.confidence
    hand_visible_time = hand.visible_time
    hand_pinch_distance = hand.pinch_distance
    hand_grab_angle = hand.grab_angle
    hand_pinch_strength = hand.pinch_strength
    hand_grab_strength = hand.grab_strength

    data = [
        # hand_id,
        # hand_flags,
        # hand_type,
        hand_confidence,
        # hand_visible_time,
        hand_pinch_distance,
        hand_grab_angle,
        hand_pinch_strength,
        hand_grab_strength,
    ]

    return data


def extract_event_data(event: TrackingEvent):
    for hand in event.hands:
        # Ignore left hand (for now)
        if hand.type == HandType.Left:
            continue

        # Extract hand data
        hand_data = extract_hand_data(hand)

        # Extract palm data
        palm_data = extract_palm_data(hand.palm)

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
                    data = torch.tensor(data)
                    print(data, len(data))
    except Exception as e:
        raise e
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
