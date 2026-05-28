#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from utils.gesture_command import (
    CMD_NONE,
    CMD_RESET_VIEW,
    GE_ACTIONS,
    CommandStabilizer,
    GestureCommandMapper,
    UX_IDLE,
    command_payload,
)
from utils.simple_gesture import classify as classify_simple_gesture
from model import KeyPointClassifier
from model import PointHistoryClassifier


WINDOW_NAME = 'Hand Gesture Recognition'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument(
        '--control-earth',
        action='store_true',
        help='Send stable gestures to Google Earth Pro via keyboard shortcuts',
    )
    parser.add_argument(
        '--earth-interval-ms',
        type=int,
        default=50,
        help='Repeat interval for pan/zoom/rotate while gesture is held (default: 50)',
    )
    parser.add_argument(
        '--earth-pan-taps',
        type=int,
        default=5,
        help='Fallback arrow-key taps sent for each pan tick (default: 5)',
    )
    parser.add_argument(
        '--earth-pan-drag-px',
        type=int,
        default=130,
        help='Mouse drag distance for each pan tick (default: 130)',
    )
    parser.add_argument(
        '--earth-horizontal-pan-drag-px',
        type=int,
        default=50,
        help='Mouse drag distance for left/right pan ticks (default: 50)',
    )
    parser.add_argument(
        '--earth-zoom-scroll-steps',
        type=int,
        default=4,
        help='Mouse wheel steps for each zoom tick (default: 4)',
    )
    parser.add_argument(
        '--earth-focus',
        action='store_true',
        help='Focus Google Earth and click the 3D map viewport before sending keys (Windows)',
    )
    parser.add_argument(
        '--earth-layout',
        action='store_true',
        help='Place the camera window on the left and Google Earth on the right',
    )
    parser.add_argument(
        '--earth-window-title',
        default='Google Earth',
        help='Window title substring used to find Google Earth (default: Google Earth)',
    )
    parser.add_argument(
        '--earth-allow-close',
        action='store_true',
        help='Allow CLOSE gesture to send Alt+F4 to Google Earth (off by default)',
    )

    # Unity integration
    parser.add_argument(
        '--control-unity',
        action='store_true',
        help='Send gesture payloads to a Unity app via UDP',
    )
    parser.add_argument(
        '--unity-host',
        default='127.0.0.1',
        help='UDP host for Unity receiver (default: 127.0.0.1)',
    )
    parser.add_argument(
        '--unity-port',
        type=int,
        default=7777,
        help='UDP port for Unity receiver (default: 7777)',
    )

    # Landmark streaming — Phase 1 hand skeleton digital twin
    parser.add_argument(
        '--stream-landmarks',
        action='store_true',
        help='Stream raw 21-landmark positions to Unity for hand skeleton visualization',
    )
    parser.add_argument(
        '--landmark-port',
        type=int,
        default=7778,
        help='UDP port for landmark streaming to Unity (default: 7778)',
    )

    parser.add_argument(
        '--preset',
        choices=['demo-smooth', 'demo-fast', 'debug'],
        default=None,
        help=(
            'Named configuration preset: '
            'demo-smooth (high confidence, gentle pan/zoom), '
            'demo-fast (quick response, larger movements), '
            'debug (verbose terminal output)'
        ),
    )

    args = parser.parse_args()

    # Apply preset overrides for earth navigation params
    if args.preset == 'demo-smooth':
        args.earth_interval_ms = 55
        args.earth_pan_drag_px = 25
        args.earth_horizontal_pan_drag_px = 12
        args.earth_zoom_scroll_steps = 5
    elif args.preset == 'demo-fast':
        args.earth_interval_ms = 20
        args.earth_pan_drag_px = 180
        args.earth_horizontal_pan_drag_px = 70
        args.earth_zoom_scroll_steps = 5

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    if not cap.isOpened():
        print(
            f"Error: Could not open camera device {cap_device}. "
            "Check that a webcam is connected and not used by another app."
        )
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    if args.earth_layout:
        from utils.earth_bridge import arrange_google_earth_window

        camera_rect = arrange_google_earth_window(args.earth_window_title)
        if camera_rect is None:
            print(
                f'Warning: Could not find a window containing '
                f'"{args.earth_window_title}". Open Google Earth Pro first.'
            )
        else:
            x, y, w, h = camera_rect
            cv.resizeWindow(WINDOW_NAME, w, min(h, cap_height + 180))
            cv.moveWindow(WINDOW_NAME, x, y)
            print(
                'Earth layout enabled: camera window on the left, '
                'Google Earth window on the right.'
            )

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    # Gesture command layer ##################################################
    if args.preset == 'demo-smooth':
        _stab_buf, _stab_votes, _stab_hold = 12, 7, 35
        _flicker_threshold = 8
    elif args.preset == 'demo-fast':
        _stab_buf, _stab_votes, _stab_hold = 7, 4, 20
        _flicker_threshold = 4
    else:
        _stab_buf, _stab_votes, _stab_hold = 9, 5, 28
        _flicker_threshold = 6

    verbose = (args.preset == 'debug')
    command_mapper = GestureCommandMapper(nav_flicker_threshold=_flicker_threshold)
    command_stabilizer = CommandStabilizer(
        buffer_len=_stab_buf,
        min_votes=_stab_votes,
        hold_frames=_stab_hold,
    )
    active_payload = command_payload(CMD_NONE, 0.0, "init", ux_state=UX_IDLE)
    last_printed_command = CMD_NONE

    unity_bridge = None
    if args.control_unity:
        from utils.unity_bridge import UnityBridge

        unity_bridge = UnityBridge(host=args.unity_host, port=args.unity_port)
        print(
            f"Unity control enabled. Sending UDP payloads to "
            f"{args.unity_host}:{args.unity_port}"
        )

    landmark_bridge = None
    if args.stream_landmarks:
        from utils.landmark_bridge import LandmarkBridge

        landmark_bridge = LandmarkBridge(host=args.unity_host, port=args.landmark_port)
        print(
            f"Landmark streaming enabled. Sending to "
            f"{args.unity_host}:{args.landmark_port}"
        )

    earth_bridge = None
    if args.control_earth:
        from utils.earth_bridge import EarthBridge

        earth_bridge = EarthBridge(
            interval_ms=args.earth_interval_ms,
            auto_focus=args.earth_focus,
            window_title=args.earth_window_title,
            allow_close=args.earth_allow_close,
            pan_taps=args.earth_pan_taps,
            pan_drag_px=args.earth_pan_drag_px,
            horizontal_pan_drag_px=args.earth_horizontal_pan_drag_px,
            zoom_scroll_steps=args.earth_zoom_scroll_steps,
        )
        print(
            "Google Earth control enabled. "
            "Click the Google Earth window (or use --earth-focus), then use gestures. "
            "Pan/zoom/rotate repeat while held; R still resets from keyboard."
        )

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        keyboard_reset = key in (ord('r'), ord('R'))
        if keyboard_reset:
            command_mapper.request_reset()
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        raw_command = CMD_NONE
        raw_source = "no_hand"
        hand_sign_text = ""
        finger_gesture_text = ""
        # Landmark-stream state — populated inside the detection loop
        _lm_cache = None
        _simple_cmd = "IDLE"
        _index_x = 0.0
        _index_y = 0.0

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                hand_sign_text = keypoint_classifier_labels[hand_sign_id]
                finger_gesture_text = point_history_classifier_labels[
                    most_common_fg_id[0][0]]

                raw_command, raw_source = command_mapper.map_command(
                    hand_sign_text,
                    finger_gesture_text,
                    brect,
                    debug_image.shape,
                    point_history,
                    landmark_list,
                )

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    hand_sign_text,
                    finger_gesture_text,
                )

                if landmark_bridge is not None:
                    # Classify with the robust distance-based rule
                    # (rotation-invariant, independent of the trained classifier)
                    _simple_cmd = classify_simple_gesture(hand_landmarks.landmark)
                    _index_x = float(hand_landmarks.landmark[8].x)
                    _index_y = float(hand_landmarks.landmark[8].y)
                    _lm_cache = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        else:
            point_history.append([0, 0])
            command_mapper.reset_hand_tracking()

        if keyboard_reset:
            raw_command = CMD_RESET_VIEW
            raw_source = "keyboard_reset"
            for _ in range(command_stabilizer.buffer.maxlen):
                command_stabilizer.update(CMD_RESET_VIEW)

        stable_command, confidence = command_stabilizer.update(raw_command)

        # Send the landmark packet with the SIMPLE classifier output
        # (distinct from stable_command; the simple one is rotation-invariant
        # and is what the Unity controller drives the camera from).
        if landmark_bridge is not None:
            if _lm_cache is not None:
                landmark_bridge.send(
                    _lm_cache,
                    command=_simple_cmd,
                    index_x=_index_x,
                    index_y=_index_y,
                )
            else:
                landmark_bridge.send_no_hand()

            # Debug print: every frame, exactly what Unity receives
            print(f"UDP SEND: {_simple_cmd} | Index: {_index_x:.2f}, {_index_y:.2f}")

        active_payload = command_payload(
            stable_command,
            confidence,
            raw_source,
            hand_sign=hand_sign_text,
            finger_gesture=finger_gesture_text,
            ux_state=command_mapper.ux_state,
        )

        # Old trained-classifier prints disabled — the new UDP SEND line
        # below is the single source of truth for what Unity sees.
        # (Re-enable by uncommenting if you need to debug the old pipeline.)
        # if verbose and raw_command != CMD_NONE:
        #     print(f"[RAW] {raw_command} src={raw_source} ux={command_mapper.ux_state}")
        # if stable_command != last_printed_command and stable_command != CMD_NONE:
        #     print(f"[COMMAND] {stable_command} ({GE_ACTIONS.get(stable_command, '')}) "
        #           f"conf={confidence:.0%} src={raw_source}")
        #     last_printed_command = stable_command
        # elif stable_command == CMD_NONE:
        #     last_printed_command = CMD_NONE

        if earth_bridge is not None:
            earth_bridge.tick(stable_command)
            active_payload["earth_status"] = earth_bridge.last_status

        if unity_bridge is not None:
            unity_bridge.tick(active_payload)

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        debug_image = draw_command_panel(
            debug_image,
            active_payload,
            earth_control_enabled=earth_bridge is not None,
        )

        # Screen reflection #############################################################
        cv.imshow(WINDOW_NAME, debug_image)

    cap.release()
    cv.destroyAllWindows()
    if unity_bridge is not None:
        unity_bridge.close()
    if landmark_bridge is not None:
        landmark_bridge.close()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_command_panel(image, payload, earth_control_enabled=False):
    h, w = image.shape[:2]
    panel_h = 132 if earth_control_enabled else 118
    cv.rectangle(image, (0, h - panel_h), (w, h), (32, 32, 32), -1)

    command = payload.get("command", CMD_NONE)
    conf = payload.get("confidence", 0.0)
    earth = payload.get("earth_action", "")
    hand_sign = payload.get("hand_sign", "")
    finger = payload.get("finger_gesture", "")
    earth_status = payload.get("earth_status", "idle")
    ux_state = payload.get("ux_state", "IDLE")

    # Line 1: Command (left) + UX State (right)
    cmd_color = (80, 220, 120) if command != CMD_NONE else (180, 180, 180)
    cv.putText(image, f"Command: {command}", (12, h - panel_h + 28),
               cv.FONT_HERSHEY_SIMPLEX, 0.85, cmd_color, 2, cv.LINE_AA)
    cv.putText(image, f"State: {ux_state}", (w // 2, h - panel_h + 28),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 210), 1, cv.LINE_AA)

    # Line 2: Earth action
    cv.putText(image, f"Action: {earth}", (12, h - panel_h + 52),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv.LINE_AA)

    # Line 3: Gesture + Confidence
    gesture_text = f"Gesture: {hand_sign}"
    if finger:
        gesture_text += f" + {finger}"
    gesture_text += f"   Confidence: {conf:.0%}"
    cv.putText(image, gesture_text, (12, h - panel_h + 76),
               cv.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv.LINE_AA)

    # Line 4: Hint
    cv.putText(
        image,
        "Open=pan  Closed=zoom in  OK=zoom out  Stop hold=stop  R=reset",
        (12, h - panel_h + 98),
        cv.FONT_HERSHEY_SIMPLEX,
        0.38,
        (160, 160, 160),
        1,
        cv.LINE_AA,
    )

    # Line 5: Earth control status (if enabled)
    if earth_control_enabled:
        _status_colors = {
            "driving": (100, 200, 255),
            "stopped": (100, 255, 180),
            "applied": (100, 255, 180),
            "blocked": (80, 80, 200),
        }
        _status_labels = {
            "idle": "idle",
            "waiting": "waiting",
            "driving": "driving",
            "stopped": "stopped",
            "applied": "applied",
            "blocked": "blocked (CLOSE disabled)",
        }
        ge_label = _status_labels.get(earth_status, earth_status)
        ge_color = _status_colors.get(earth_status, (180, 180, 180))
        cv.putText(
            image,
            f"Control: {ge_label}",
            (12, h - panel_h + 118),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            ge_color,
            1,
            cv.LINE_AA,
        )
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
