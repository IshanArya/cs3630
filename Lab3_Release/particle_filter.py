# Ishan Arya & Seung Je Jung

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
    (dx, dy, dh) = odom
    if dx == 0 and dy == 0 and dh == 0:
        return particles
    for particle in particles:
        noisy_odom = add_odometry_noise(
            odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)
        dx, dy, dh = noisy_odom
        x = particle.x
        y = particle.y
        a = particle.h
        globalDx, globalDy = rotate_point(dx, dy, a)
        newX = x + globalDx
        newY = y + globalDy
        motion_particles.append(Particle(newX, newY, (a + dh) % 360))

    return motion_particles

# ------------------------------------------------------------------------


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

    weights = []
    outOfGrid = 0
    if len(measured_marker_list):
        for particle in particles:
            particleMarkers = particle.read_markers(grid)
            if not len(particleMarkers):
                weights.append(0)
            elif not grid.is_free(particle.x, particle.y):
                weights.append(0)
                outOfGrid += 1
            else:
                currentWeight = 1
                for robotMarker in measured_marker_list:
                    rmX, rmY, rmH = robotMarker
                    closestParticleMarker = min(
                        particleMarkers, key=lambda pm: grid_distance(rmX, rmY, pm[0], pm[1]))

                    pmX, pmY, pmH = closestParticleMarker
                    minDistance = grid_distance(rmX, rmY, pmX, pmY)
                    diffAngle = diff_heading_deg(rmH, pmH)

                    power = ((minDistance ** 2)/(2 * (MARKER_TRANS_SIGMA ** 2))) + \
                        ((diffAngle ** 2)/(2 * (MARKER_ROT_SIGMA ** 2)))
                    currentWeight *= math.exp(-power)

                # markerDifference = abs(len(particleMarkers) -
                #                        len(measured_marker_list))

                # if len(particleMarkers) > len(measured_marker_list):
                #     scale = DETECTION_FAILURE_RATE ** markerDifference
                #     currentWeight *= scale
                # elif len(particleMarkers) < len(measured_marker_list):
                #     scale = SPURIOUS_DETECTION_RATE ** markerDifference
                #     currentWeight *= scale
                weights.append(currentWeight)

    sumOfWeights = sum(weights)

    if sumOfWeights > 0 and len(measured_marker_list) >= 1:
        weights = list(map(lambda x: x / sumOfWeights, weights))
    else:
        weights = [1 / len(particles)] * len(particles)

    # resampling
    measured_particles = [Particle(p.x, p.y, p.h) for p in np.random.choice(
        particles, size=(len(particles) - outOfGrid), p=weights)]
    for i in range(outOfGrid):
        rX, rY = grid.random_free_place()
        measured_particles.append(Particle(rX, rY))

    return measured_particles
