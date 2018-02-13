import sys, pygame, math, random, time, copy, operator
import numpy as np
from pygame.locals import *
import pandas as pd
import pdb
from readmap import *
import re
import logging
from numpy import ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timeit

# General parameters
NUM_INITIAL_SAMPLES = 100
LOG_FILENAME = 'data/log/robotdata3.log'    # Only work with robotdata1 or robotdata3

# Error parameters for measurement model
THETA_HIT = 100  # standard deviation of gaussian
LAMBDA_SHORT = 0.0001  # exponential distribution
Z_MAX = 8183  # max range of laser readings (point-mass distribution)
COEF = [0.7, 0.2, 0.1, 0.1]  # 4 values to linearly combine the distributions (add to one)
SMALL_PROB = 0.000001  # If weights are zero set to this value to avoid errors

"""
Import data as pandas df
Export as numpy array
"""


def import_log_data(log):

    col_names = ["Type"] + ["col"] * 187
    data = pd.read_csv(log, delimiter=' ', names=col_names, engine='python')
    data = data[1:]
    odometer_data = data.loc[data['Type'] == "O"]
    odometer_data = odometer_data.iloc[:, 1:5]
    odometer_data = odometer_data.values
    laser_data = data.loc[data['Type'] == "L"]
    laser_data = laser_data.iloc[:, 1:]
    laser_data = laser_data.values
    data = data.values
    return odometer_data, laser_data, data

def motion_model(particle_list, odom_tminus1, odom_t):

    # Extract motion by finding difference in odometer data
    delta_x = odom_t[0] - odom_tminus1[0]
    delta_y = odom_t[1] - odom_tminus1[1]
    
    # Calculate the change in motion  
    delta_rot1 = math.atan2(delta_y, delta_x) - odom_tminus1[2]
    delta_trans = math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2))
    delta_rot2 = odom_t[2] - odom_tminus1[2] - delta_rot1

    # Apply the controls to each particle
    for particle in particle_list:
        # Add noise to the motion
        delta_rot1 = delta_rot1 - np.random.normal(0, abs(delta_rot1) * A1 + A2 * abs(delta_trans) + .001)
        delta_trans = delta_trans - np.random.normal(0, abs(delta_trans) * A3 + A4 * abs(delta_rot1 + delta_rot2) + .01)
        delta_rot2 = delta_rot2 - np.random.normal(0, abs(delta_rot2) * A1 + A2 * abs(delta_trans) + .001)
        particle[0] = particle[0] + delta_trans * math.cos(particle[2] + delta_rot1)
        particle[1] = particle[1] + delta_trans * math.sin(particle[2] + delta_rot1)
        particle[2] = particle[2] + delta_rot1 + delta_rot2


"""
Apply measurement_model
observation: x y theta xl yl thetal r1 ... r180 ts
position: x y theta
n: number of samples to take from observation
map: map
"""

def measurement_model(observation, particle_list, n, map):
    step = int(180/n)
    weights = []
    for particle in particle_list:
        ground_truth = ray_tracing(particle, n, map, math.degrees(observation[5] - particle[2]))
        observed = np.array([observation[i] for i in range(0, 180, step)])  # sample the observations
        probability = 1
        for i in range(n):
            probability *= get_prob(ground_truth[i], observed[i])
        weights.append(probability)

    if sum(weights) == 0:  # assign a small probability to avoid zero errors
        logging.warn('All weights are zero - setting it to a small value to avoid errors')
        weights = [SMALL_PROB] * n
    return weights


# Get probability of an observation based on the distribution - linear combination of 4 prob dist
def get_prob(expected, observed):

    uniform = 1/Z_MAX

    point_mass = 1 if observed == Z_MAX else 0

    const = 1 / (math.sqrt(2 * math.pi) * THETA_HIT)
    gaussian = const * math.exp(-0.5 * math.pow((observed - expected)/THETA_HIT, 2))

    exp = 0
    if 0 <= observed <= expected:  # TODO(Ishita): Check here for values above 1
        c = 1 / (1 - math.exp(-1 * LAMBDA_SHORT * expected))
        exp = c * LAMBDA_SHORT * math.exp(-1 * LAMBDA_SHORT * observed)

    if exp > 1:
        print('cd= %f' % (1 - math.exp(-1 * LAMBDA_SHORT * expected)))
        print('p=%f' % (LAMBDA_SHORT * math.exp(-1 * LAMBDA_SHORT * observed)))
        print('exp: %f' % exp)

    weight = COEF[0] * gaussian + COEF[1] * exp + COEF[2] * uniform + COEF[3] * point_mass
    return weight

"""
observation: x y theta xl yl thetal r1 ... r180 ts
position: O x y theta ts
"""


# Return np array of expected distances from position (sampling at n angles from 0-180)
def ray_tracing(position, n, map, theta_offset):
    result = []
    step = int(180 / n)

    # Generate n rays (all points on ray)
    # Adjust x and y by 25 cos theta1
    x = (position[0] * map.meta['resolution'] + 25 * math.cos(theta_offset)) / map.meta['resolution']
    y = (position[1] * map.meta['resolution'] + 25 * math.cos(theta_offset)) / map.meta['resolution']

    # Find all angles
    angles = np.array([i + theta_offset for i in range(0, 180, step)])
    cos_theta = np.cos(np.radians(angles))
    sin_theta = np.sin(np.radians(angles))

    # x = x + 1 * cos theta , 2*cos theta ....
    # Therefore every row corresponds to a specific ray
    all_x = np.add(x, np.outer(cos_theta, np.arange(0, map.meta['rows'])))
    all_y = np.add(y, np.outer(sin_theta, np.arange(0, map.meta['cols'])))

    for row_num in range(len(all_x)):
        curr_x = all_x[row_num, :]
        curr_y = all_y[row_num, :]

        # Find minimum point where the ray intersects (prob <= 0)
        # Since x and y may be out of index (800), filter those x and y coordinates
        coordinates = np.array(list(zip(curr_x, curr_y)))
        mask = (curr_x < map.meta['rows']) & (curr_y < map.meta['cols']) & (curr_x >= 0) & (curr_y >= 0)
        coordinates = coordinates[mask, :].astype(int)
        list_prob = map.map[coordinates[:, 0], coordinates[:, 1]]
        rows = np.where(list_prob <= 0.1)  # Not zero - to adjust for noise
        rows = rows[0]

        if len(rows) == 0:  # No point of intersection found - set to max value
            result.append(Z_MAX)
            continue

        point_of_intersection = coordinates[rows[0], :]
        # Find distance to the point
        d = math.sqrt(
            math.pow(point_of_intersection[0] * 10 - x * 10, 2) + math.pow(point_of_intersection[1] * 10 - y * 10,
                                                                           2))
        result.append(d)
    return result


"""
Low-variance sampler
To reduce sampling error
M: number of samples we want to resample
"""


def low_variance_sampler(M, particle_list, weights):
    logging.info('Re-sampling particles')
    particle_list_new = []
    r = random.random()* 1/M
    c = weights[0]
    i = 0

    for m in range(M):
        u = r+float(m)/M
        while u > c and i < len(weights):
            i += 1
            c = c+weights[i]
        if i < len(particle_list):
            particle_list_new.append(particle_list[i])

    return particle_list_new


def main():
    # Timer
    start = timeit.default_timer()

    # Setup logging to track program
    log_file = 'logs.txt'
    logging.basicConfig(filename=log_file, level=logging.DEBUG)

    # Read in map
    map_filename = 'data/map/wean.dat'
    map1 = Map(map_filename)

    # Read in data
    odometer_data, laser_data, data = import_log_data(LOG_FILENAME)
    logging.info('Finished reading map and data')

    # Parameters for initialization
    map_numcols = map1.meta["global_mapsize_x"]/map1.meta["resolution"]
    map_numrows = map1.meta["global_mapsize_y"]/map1.meta["resolution"]
    map_cells = map_numrows * map_numcols

    # Parameters for motion model
    global A1, A2, A3, A4  # pctg of translation and rotation that are added to model noise

    theta_diff = np.diff(odometer_data[:,2])
    theta_diff_sd = round(np.std(theta_diff, 0),1)
    A4 = 0.1*theta_diff_sd  # noise on drot1+drot2
    A1 = 0.05*theta_diff_sd # noise on drot1 and drot2

    x_diff = np.diff(odometer_data[:, 0])
    y_diff = np.diff(odometer_data[:, 1])
    xy_diff2 = x_diff*x_diff + y_diff*y_diff
    xy_diff2_sqrt = np.sqrt(xy_diff2)
    xy_diff_sd = round(np.std(xy_diff2_sqrt,0),1)
    A2 = 0.1*xy_diff_sd    # noise on translation for drot
    A3 = 0.2*xy_diff_sd    # noise on translation for dtrans
    #SD_XY = (math.ceil(SD_XY * 10) / 10)    # round up to tenth decimal place
    A1 = .075

    logging.info('Parameter initialization complete')

    # Initialize a list of particles defined by x,y,theta
    map2 = np.copy(map1.map)
    map2[map2 < 0] = 0
    map2 = np.reshape(map2, map_cells)
    map2 = np.array(map2)/float(sum(map2))
    samples = np.random.choice(map_cells, NUM_INITIAL_SAMPLES, replace=True, p=map2)

    # Sample assuming you can only put one particle per cell
    particle_list = np.array([[i%map_numrows, i/map_numcols, random.random()*2*math.pi-math.pi] for i in samples])
    logging.info('Initialized particle list - Starting particle filter')

    # Particle filter
    curr_i = 0
    num_resamples = NUM_INITIAL_SAMPLES

    while curr_i < len(data):   #TODO: look at the vis file- the points aren't converging :(
        logging.info('Iteration %d' % curr_i)

        curr_laser_data =[]
        curr_odometer_data = []

        while data[curr_i][0] != 'L' and curr_i < len(data)-1:
            curr_odometer_data.append(data[curr_i])
            curr_i+=1
        if data[curr_i][0] == 'L' and curr_i < len(data):
            curr_laser_data.append(data[curr_i])
        if (len(curr_odometer_data) != 0):
            curr_odometer_data = np.array(curr_odometer_data)[:,1:5]
            curr_laser_data = np.array(curr_laser_data)[:,1:]

        if len(curr_odometer_data) != 0 and (curr_odometer_data[0] != curr_odometer_data[len(curr_odometer_data)-1]).any():  # Maintain variance if robot is static

            # Motion Model
            motion_model(particle_list, curr_odometer_data[0], curr_odometer_data[len(curr_odometer_data)-1])
            logging.info('Motion model applied')
            #logging.info('Motion modeled particle data %s' % str(particle_list))

            # Sensor Model
            weights = measurement_model(curr_laser_data[0], particle_list, 4, map1)
            logging.info('Sensor model applied')
            #logging.info('Probabiliy weights %s' % str(weights))

            # if curr_i % 10 == 0:
            #     plt.imshow(map1.map)
            #     # if len(weights) > 100:
            #     #     ind = np.argpartition(weights, -100)[-100:]
            #     # else:
            #     #     ind = np.arange(0,len(weights) - 1)
            #     ind = np.arange(0,len(weights) - 1)
            #     for i in ind:
            #         particle = particle_list[i]
            #         plt.plot([particle[0]], [particle[1]], 'wo')
            #     #plt.show()
            #     plt.savefig('Vis/iteration_{}'.format(curr_i))
            #     logging.info('save fig')

            logging.info('About to normalize weights')
            weights = np.array(weights)/sum(weights)    # TODO: weights are all super small?

            # Resample weights
            if 1 <= curr_i <= 40:
                num_resamples = int(num_resamples * 0.95)
            else:
                num_resamples = int(NUM_INITIAL_SAMPLES / 10)
            #logging.info('Number of resamples are %d' % num_resamples)
            #particle_indexes = np.array(np.random.choice(len(particle_list), num_resamples, True, weights))
            #particle_list = particle_list[particle_indexes]
            particle_list = low_variance_sampler(num_resamples, particle_list, weights)
            particle_list = np.array(particle_list)
            #logging.info('Resampled particle data %s' % str(particle_list))
        else:
            logging.warn('Robot is static')
        curr_i += 1

    localized_position = np.mean(particle_list, 0)
    print "reached the end", localized_position
    # make the orientation in range [-pi,pi]
    localized_position[2] = ((localized_position[2] + math.pi) % (math.pi*2)) - math.pi
    logging.info('Completed!')
    logging.info('Final pos: %d %d %d' % (localized_position[0], localized_position[1], localized_position[2]))
    stop = timeit.default_timer()
    print "time it took to run this program %s" %(stop - start)
    return localized_position


# Run the program
if __name__ == '__main__':
    main()

