import lab1
import pylab as plt
import numpy as np
import math

def test_measurement_model():
    map = lab1.Map('data/map/wean.dat')
    particle_list = [[400, 400, 0]]  # list of: x y theta

    _, laser_data = lab1.import_log_data('data/log/robotdata1.log')
    observation = laser_data[0]
    n = 4

    # Run whole measurement model test
    weights = lab1.measurement_model(observation, particle_list, n, map)
    print (weights)

    # Sum should not be 0
    assert sum(weights) != 0

    # Len should be equal to particle list length
    assert len(weights) == len(particle_list)

    # All weights should be less than 1 (since they are probabilities)
    assert all([weight <= 1 for weight in weights])

    print ('Test measurement model success')

def test_get_prob():
    observed = 5
    expected = 6
    prob = lab1.get_prob(observed=observed, expected=expected)

    # Probability should be less than 1
    assert 0 <= prob <= 1
    print ('Test prob success')


def test_ray_tracing():
    map = lab1.Map('data/map/wean.dat')
    im = plt.imshow(map.map * 255)
    plt.colorbar(im, orientation='horizontal')

    position = [398.2, 463.2, 0]
    plt.plot([position[0]], [position[1]], 'r,')

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
            mask = (curr_x < 800) & (curr_y < 800) & (curr_x >= 0) & (curr_y >= 0)
            coordinates = coordinates[mask, :].astype(int)
            plt.plot(coordinates[:, 0], coordinates[:, 1], 'b,')
            list_prob = map.map[coordinates[:, 0], coordinates[:, 1]]
            rows = np.where(list_prob <= 0.1)  # Not zero - to adjust for noise
            rows = rows[0]

            if len(rows) == 0:  # No point of intersection found - return max value
                result.append(8173)
                continue

            point_of_intersection = coordinates[rows[0], :]
            plt.plot([point_of_intersection[0]], [point_of_intersection[1]], 'r+')

            # Find distance to the point
            d = math.sqrt(
                math.pow(point_of_intersection[0] * 10 - x * 10, 2) + math.pow(point_of_intersection[1] * 10 - y * 10,
                                                                               2))
            result.append(d)
        return result

    ray_tracing(position, 4, map, 0)
    plt.show()
    print ('Test ray tracing success')

if __name__ == "__main__":
    test_get_prob()
    test_measurement_model()
    test_ray_tracing()
