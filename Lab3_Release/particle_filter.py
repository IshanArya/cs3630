#Ishan Arya
#Seung Je Jung

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
        # noisy_odom = add_odometry_noise(
        #     odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)
        # dx, dy, dh = noisy_odom
        x = particle.x
        y = particle.y
        a = particle.h
        globalDx, globalDy = rotate_point(dx, dy, a)
        newX = x + globalDx
        newY = y + globalDy
        motion_particles.append(Particle(newX, newY, (a + dh) % 360))

    return motion_particles

# ------------------------------------------------------------------------


def changeBasis(particle, relativeCoord):
    rotX, rotY = rotate_point(relativeCoord[0], relativeCoord[1], particle.h)
    globX = rotX + particle.x
    globY = rotY + particle.y
    return (globX, globY)

def calcNormalDist(mean, stdev, point):
    pz = (point + stdev - mean) / stdev
    nz = (point - stdev - mean) / stdev
    return 0.5 * math.erf(pz / (2**0.5)) - 0.5 * math.erf(nz / (2**0.5))


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
            mml = [x for x in measured_marker_list]
            while len(mml) != 0 and len(particleMarkers) != 0:
                currBestPair = (-1, -1)
                currBestDist = float("inf")
                for i, measured_marker in enumerate(mml):
                    mX, mY, mH = measured_marker
                    for (j, particleMarker) in enumerate(particleMarkers):
                        px, py, ph = particleMarker
                        if currBestDist > grid_distance(px, py, mX, mY):
                            currBestDist = grid_distance(px, py, mX, mY)
                            currBestPair = (i, j)

                mX, mY, mH = mml[currBestPair[0]]
                px, py, ph = particleMarkers[currBestPair[1]]
                diffAngle = abs(ph - mH)
                power = ((currBestDist ** 2)/(2 * (MARKER_TRANS_SIGMA ** 2))) + \
                    (((diffAngle) ** 2)/(2 * (MARKER_ROT_SIGMA ** 2)))
                weights[i] *= math.exp(-power)

                mml.pop(currBestPair[0])
                particleMarkers.pop(currBestPair[1])

            
    #print(weights)
    sumOfWeights = sum(weights)

    if sumOfWeights != 0:
        weights = list(map(lambda x: x / sumOfWeights, weights))
    else:
        weights = [1 / len(particles)] * len(particles)
    # TODO resampling
    measured_particles = np.random.choice(particles, len(particles), replace=True, p=weights)
    return measured_particles
