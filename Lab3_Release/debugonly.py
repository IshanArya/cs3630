from particle import *
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
        motion_particles.append(Particle(newX, newY, a + dh))

    return motion_particles

particle = Particle(0, 0, -60)
print(motion_update([particle], (2, 0, 0)))