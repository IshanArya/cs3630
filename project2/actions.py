from cozmo.util import *
import cozmo

def cozmo_program(robot: cozmo.robot.Robot):
	"""
	#Drone

	cube = robot.world.wait_for_observed_light_cube()
	action = robot.pickup_object(cube)
	action.wait_for_completed()
	action = robot.drive_straight(Distance(distance_mm = 100), Speed(speed_mmps = 30))
	action.wait_for_completed()
	action = robot.place_object_on_ground_here(cube)
	action.wait_for_completed()
	action = robot.drive_straight(Distance(distance_mm = -100), Speed(speed_mmps = 30))
	action.wait_for_completed()"""

	"""
	#Order

	action = robot.drive_wheels(l_wheel_speed = 180, r_wheel_speed = 80, duration = 6)
	action.wait_for_completed()"""

	"""
	#Inspection

	global liftLocation
	global raiseA
	liftLocation = 1
	raiseAction = robot.set_lift_height(height = 1.0, duration = 2.0, in_parallel = True)
	def changeLiftHeight(*args, **kwargs):
		global liftLocation
		global raiseA
		liftLocation = 1 - liftLocation
		raiseA = robot.set_lift_height(height = liftLocation, duration = 2.0, in_parallel = True)
		raiseA.on_completed(changeLiftHeight)
	raiseAction.on_completed(changeLiftHeight)

	robot.drive_straight(Distance(distance_mm = 100), Speed(speed_mmps = 30), in_parallel = True).wait_for_completed()
	robot.turn_in_place(Angle(degrees = 90), in_parallel = True).wait_for_completed()
	robot.drive_straight(Distance(distance_mm = 100), Speed(speed_mmps = 30), in_parallel = True).wait_for_completed()
	robot.turn_in_place(Angle(degrees = 90), in_parallel = True).wait_for_completed()
	robot.drive_straight(Distance(distance_mm = 100), Speed(speed_mmps = 30), in_parallel = True).wait_for_completed()
	robot.turn_in_place(Angle(degrees = 90), in_parallel = True).wait_for_completed()
	robot.drive_straight(Distance(distance_mm = 100), Speed(speed_mmps = 30), in_parallel = True).wait_for_completed()
	robot.turn_in_place(Angle(degrees = 90), in_parallel = True).wait_for_completed()
	if raiseA != None:
		raiseA.abort()"""

cozmo.run_program(cozmo_program)