from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np
import math
import random
np.random.seed(setting.RANDOM_SEED)
random.seed(setting.RANDOM_SEED)


def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """

    motion_particles = []
    print(particles)
    (dx, dy, dh) = odom
    if dx == 0 and dy == 0 and dh == 0:
        return particles
    rad = -math.radians(dh)
    for particle in particles:
        # noisy_odom = add_odometry_noise(
        #     odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)
        # dx, dy, dh = noisy_odom
        x = particle.x
        y = particle.y
        a = particle.h
        globalDx, globalDy = rotate_point(dx, dy, a)
        newX = x + globalDx
        newY = y + globalDy
        motion_particles.append(Particle(newX, newY, a + dh))

    return motion_particles

# ------------------------------------------------------------------------


def changeBasis(particle, relativeCoord):
    rotX, rotY = rotate_point(relativeCoord[0], relativeCoord[1], particle.h)
    globX = rotX + particle.x
    globY = rotY + particle.y
    return (globX, globY)


def measurement_update(particles, measured_marker_list, grid):
    """ 
    Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
                                * Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []

    weights = [1.0] * len(particles)

    for i, particle in enumerate(particles):
        if not grid.is_in(particle.x, particle.y):
            weights[i] = 0
            continue
        particleMarkers = particle.read_markers(grid)

        if len(particleMarkers) == 0 and len(measured_marker_list) == 0:
            weights[i] = 1
        elif len(particleMarkers) == 0 or len(measured_marker_list) == 0:
            weights[i] = DETECTION_FAILURE_RATE * SPURIOUS_DETECTION_RATE
        else:
            for robotMarker in measured_marker_list:
                minDistance = float('inf')
                diffAngle = None

                rmX, rmY, rmH = robotMarker

                for particleMarker in particleMarkers:
                    pmX, pmY, pmH = particleMarker
                    currentDistance = grid_distance(rmX, rmY, pmX, pmY)
                    if currentDistance < minDistance:
                        minDistance = currentDistance
                        diffAngle = diff_heading_deg(rmH, pmH)

                power = ((minDistance ** 2)/(2 * (MARKER_TRANS_SIGMA ** 2))) + \
                    ((diffAngle ** 2)/(2 * (MARKER_ROT_SIGMA ** 2)))
                weights[i] *= math.exp(power)

        markerDifference = abs(len(particleMarkers) -
                               len(measured_marker_list))

        if len(particleMarkers) > len(measured_marker_list):
            scale = DETECTION_FAILURE_RATE ** markerDifference
            weights[i] *= scale
        elif len(particleMarkers) < len(measured_marker_list):
            scale = SPURIOUS_DETECTION_RATE ** markerDifference
            weights[i] *= scale

    sumOfWeights = sum(weights)
    weights = map(lambda x: x / sumOfWeights, weights)

    # TODO resampling

    return measured_particles
