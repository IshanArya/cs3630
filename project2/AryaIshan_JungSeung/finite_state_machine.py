import numpy as np
import imgclassification
from sklearn import metrics
from skimage import io
import cozmo
from cozmo.util import *
import time
import threading


def run(sdk_conn):
    img_clf = imgclassification.ImageClassifier("model.txt")
    robot = sdk_conn.wait_for_robot()
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()

    # robot.drive_straight(distance_mm(100), speed_mmps(50)).wait_for_completed()
    # robot.turn_in_place(degrees(90)).wait_for_completed()

    robot.set_lift_height(height=0).wait_for_completed()

    robot_state = "idle"

    while(True):
        time.sleep(2)
        if robot_state == "idle":
            robot_state = idle_state(robot, img_clf)
        elif robot_state == "drone":
            robot_state = drone_state(robot)
        elif robot_state == "order":
            robot_state = order_state(robot)
        elif robot_state == "inspection":
            robot_state = inspection_state(robot)
        else:
            robot_state = idle_state(robot, img_clf)


def idle_state(robot, img_clf):
    robot.set_lift_height(height=0, in_parallel=True).wait_for_completed()
    robot.set_head_angle(cozmo.util.degrees(
        0), in_parallel=True).wait_for_completed()
    latest_image = robot.world.latest_image
    new_image = latest_image.raw_image

    img_data = np.array(new_image)
    test_data = img_clf.extract_image_features([img_data])

    predicted_label = img_clf.predict_labels(test_data)[0]

    robot.say_text(str(predicted_label)).wait_for_completed()

    return predicted_label


def drone_state(robot):
    cube = robot.world.wait_for_observed_light_cube()
    action = robot.pickup_object(cube)
    action.wait_for_completed()
    action = robot.drive_straight(
        Distance(distance_mm=100), Speed(speed_mmps=30))
    action.wait_for_completed()
    action = robot.place_object_on_ground_here(cube)
    action.wait_for_completed()
    action = robot.drive_straight(
        Distance(distance_mm=-100), Speed(speed_mmps=30))
    action.wait_for_completed()

    return "idle"


def order_state(robot):
    robot.drive_wheels(
        l_wheel_speed=180, r_wheel_speed=80, duration=6)

    return "idle"


def inspection_state(robot):
    global liftLocation
    global stop
    stop = threading.Event()
    liftLocation = 1
    raiseAction = robot.set_lift_height(
        height=1.0, duration=2.0, in_parallel=True)

    def changeLiftHeight(*args, **kwargs):
        global liftLocation
        global stop
        if stop.is_set():
            return
        liftLocation = 1 - liftLocation
        raiseA = robot.set_lift_height(
            height=liftLocation, duration=2.0, in_parallel=True)
        raiseA.on_completed(changeLiftHeight)
    raiseAction.on_completed(changeLiftHeight)

    robot.drive_straight(Distance(distance_mm=100), Speed(
        speed_mmps=30), in_parallel=True).wait_for_completed()
    robot.turn_in_place(Angle(degrees=90),
                        in_parallel=True).wait_for_completed()
    robot.drive_straight(Distance(distance_mm=100), Speed(
        speed_mmps=30), in_parallel=True).wait_for_completed()
    robot.turn_in_place(Angle(degrees=90),
                        in_parallel=True).wait_for_completed()
    robot.drive_straight(Distance(distance_mm=100), Speed(
        speed_mmps=30), in_parallel=True).wait_for_completed()
    robot.turn_in_place(Angle(degrees=90),
                        in_parallel=True).wait_for_completed()
    robot.drive_straight(Distance(distance_mm=100), Speed(
        speed_mmps=30), in_parallel=True).wait_for_completed()
    robot.turn_in_place(Angle(degrees=90),
                        in_parallel=True).wait_for_completed()
    stop.set()
    robot.stop_all_motors()

    return "idle"


if __name__ == '__main__':
    cozmo.setup_basic_logging()

    try:
        cozmo.connect(run)
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
