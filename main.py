from PIL import Image
import numpy as np
import datetime

img = np.divide(np.asarray(Image.open("img/camera128.bmp")), 255.)

(nrow, ncol) = img.shape

print "Welcome to demo program of image edge detection using ant colony.\nPlease wait......\n"
print "START ", datetime.datetime.now().time()

for nMethod in range(1, 4):

    v = np.zeros(img.shape)
    v_norm = 0
    for rr in range(1, nrow):
        for cc in range(1, ncol):

            temp1 = np.array([[rr - 2, cc - 1],
                              [rr - 2, cc + 1],
                              [rr - 1, cc - 2],
                              [rr - 1, cc - 1],
                              [rr - 1, cc],
                              [rr - 1, cc + 1],
                              [rr - 1, cc + 2],
                              [rr, cc - 1]])

            temp2 = np.array([[rr + 2, cc + 1],
                              [rr + 2, cc - 1],
                              [rr + 1, cc + 2],
                              [rr + 1, cc + 1],
                              [rr + 1, cc],
                              [rr + 1, cc - 1],
                              [rr + 1, cc - 2],
                              [rr, cc + 1]])

            temp0 = np.nonzero((temp1[:, 0] >= 0)
                               & (temp1[:, 0] < nrow)
                               & (temp1[:, 1] >= 0)
                               & (temp1[:, 1] < ncol)
                               & (temp2[:, 0] >= 0)
                               & (temp2[:, 0] < nrow)
                               & (temp2[:, 1] >= 0)
                               & (temp2[:, 1] < ncol))

            temp11 = temp1[temp0]
            temp22 = temp2[temp0]

            temp00 = np.zeros(len(temp11))
            for kk in range(0, len(temp11)-1):
                temp00[kk] = abs(img[temp11[kk, 0], temp11[kk, 1]] - img[temp22[kk, 0], temp22[kk, 1]])

            if len(temp11) == 0:
                v[rr, cc] = 0
                v_norm = v_norm + v[rr, cc]
            else:
                _lambda = 10

                if nMethod == 1 % ord('F'):
                    temp00 = np.multiply(temp00, _lambda)
                elif nMethod == 2 % ord('Q'):
                    temp00 = np.multiply(np.power(temp00, 2), _lambda)
                elif nMethod == 3 % ord('S'):
                    temp00 = np.sin(np.divide(np.divide(np.multiply(temp00, np.pi), 2.), _lambda))
                elif nMethod == 4 % ord('W'):
                    temp00 = np.multiply(np.sin(np.divide(np.multiply(temp00, np.pi), _lambda)),
                                         np.divide(np.multiply(temp00, np.pi), _lambda))

                v[rr, cc] = np.sum(np.sum(np.power(temp00, 2)))
                v_norm = v_norm + v[rr, cc]

    v = np.divide(v, v_norm)
    v = np.multiply(v, 100)
    p = np.multiply(np.ones(img.shape), 0.0001)

    alpha = 1
    beta = 0.1
    rho = 0.1
    phi = 0.05

    ant_total_num = int(np.round(np.sqrt(nrow * ncol)))

    ant_pos_idx = np.zeros((int(ant_total_num), 2))
    dateTimeNow = datetime.datetime.now()
    date_list = [dateTimeNow.year,
                 dateTimeNow.month,
                 dateTimeNow.day,
                 dateTimeNow.hour,
                 dateTimeNow.minute,
                 dateTimeNow.second]

    np.random.seed(np.sum(date_list))
    temp = np.random.rand(ant_total_num, 2)
    ant_pos_idx[:, 0] = np.round(1 + (nrow - 1) * temp[:, 0])
    ant_pos_idx[:, 1] = np.round(1 + (ncol - 1) * temp[:, 1])

    search_clique_mode = '8'

    if nrow * ncol == 128 * 128:
        A = 40
        memory_length = round(np.multiply(np.random.rand(1), (1.15 * A - 0.85 * A) + 0.85 * A))  # memory length
    elif nrow * ncol == 256 * 256:
        A = 30
        memory_length = round(np.multiply(np.random.rand(1), (1.15 * A - 0.85 * A) + 0.85 * A))  # memory length
    elif nrow * ncol == 512 * 512:
        A = 20
        memory_length = round(np.multiply(np.random.rand(1), (1.15 * A - 0.85 * A) + 0.85 * A))  # memory length
    else:
        print "ERROR: 'memory_length' not defined at line 110."
        exit()

    # record the positions in ant's memory, convert 2D position-index (row, col) into
    # 1D position-index
    # print int(memory_length)
    ant_memory = np.zeros((ant_total_num, memory_length))

    # System setup
    if nrow * ncol == 128 * 128:
        total_step_num = 300  # the number of iterations?
    elif nrow * ncol == 256 * 256:
        total_step_num = 900
    elif nrow * ncol == 512 * 512:
        total_step_num = 1500

    total_iteration_num = 3

    for iteration_idx in range(0, total_iteration_num-1):

        # record the positions where ant have reached in the last 'memory_length' iterations
        delta_p = np.zeros((nrow, ncol))
        for step_idx in range(0, total_step_num - 1):

            delta_p_current = np.zeros((nrow, ncol))

            for ant_idx in range(0, ant_total_num-1):

                ant_current_row_idx = ant_pos_idx[ant_idx, 0]
                ant_current_col_idx = ant_pos_idx[ant_idx, 1]

                # find the neighborhood of current position
                if search_clique_mode == '4':
                    rr = ant_current_row_idx
                    cc = ant_current_col_idx
                    ant_search_range_temp = np.array([[rr - 1, cc],
                                                      [rr, cc + 1],
                                                      [rr + 1, cc],
                                                      [rr, cc - 1]])

                elif search_clique_mode == '8':
                    rr = ant_current_row_idx
                    cc = ant_current_col_idx
                    ant_search_range_temp = np.array([[rr - 1, cc - 1],
                                                      [rr - 1, cc],
                                                      [rr - 1, cc + 1],
                                                      [rr, cc - 1],
                                                      [rr, cc + 1],
                                                      [rr + 1, cc - 1],
                                                      [rr + 1, cc],
                                                      [rr + 1, cc + 1]])

                # remove the positions our of the image's range

                temp = np.nonzero((ant_search_range_temp[:, 0] >= 0)
                                  & (ant_search_range_temp[:, 0] < nrow)
                                  & (ant_search_range_temp[:, 1] >= 0)
                                  & (ant_search_range_temp[:, 1] < ncol))  # <-- Error here

                ant_search_range = ant_search_range_temp[temp]

                # calculate the transit prob. to the neighborhood of current
                # position
                ant_transit_prob_v = np.zeros((len(ant_search_range)))
                ant_transit_prob_p = np.zeros((len(ant_search_range)))
                # print ant_transit_prob_v

                for kk in range(0, len(ant_search_range)-1):

                    temp = np.array(ant_search_range[kk, 0]-1)*ncol + ant_search_range[kk, 1]

                    if ~(ant_memory[ant_idx, :] == temp).any():  # not in ant's memory
                        ant_transit_prob_v[kk] = v[ant_search_range[kk, 0], ant_search_range[kk, 1]]
                        ant_transit_prob_p[kk] = p[ant_search_range[kk, 0], ant_search_range[kk, 1]]
                    else:  # in ant's memory
                        ant_transit_prob_v[kk] = 0
                        ant_transit_prob_p[kk] = 0

                # if all neighborhood are in memory, then the permissible search range is RE-calculated.
                if (np.sum(np.sum(ant_transit_prob_v)) == 0) | (np.sum(np.sum(ant_transit_prob_p)) == 0):
                    for kk in range(0, len(ant_search_range) - 1):
                        temp = np.multiply((ant_search_range[kk, 0]-1), ncol) + ant_search_range[kk, 1]
                        ant_transit_prob_v[kk] = v[ant_search_range[kk, 0], ant_search_range[kk, 1]]
                        ant_transit_prob_p[kk] = p[ant_search_range[kk, 0], ant_search_range[kk, 1]]
                # print ant_transit_prob_p
                if np.sum(ant_transit_prob_v) == 0 and np.sum(ant_transit_prob_v) == 0:
                    ant_transit_prob = [0., 0., 0., 0., 0., 0., 0., 0.]
                else:
                    ant_transit_prob = np.divide(np.multiply(np.power(ant_transit_prob_v, alpha),
                                                             np.power(ant_transit_prob_p, beta)),
                                                 np.sum(np.sum(np.multiply(np.power(ant_transit_prob_v, alpha),
                                                                           (np.power(ant_transit_prob_p, beta))))))

                # generate a random number to determine the next position.
                dateTimeNow = datetime.datetime.now()
                date_list = [dateTimeNow.year,
                             dateTimeNow.month,
                             dateTimeNow.day,
                             dateTimeNow.hour,
                             dateTimeNow.minute,
                             dateTimeNow.second]

                np.random.seed(np.sum(np.multiply(date_list, 100)))
                t = np.random.rand(1)
                t = np.array(np.nonzero(np.cumsum(ant_transit_prob) >= t))
                t = t[0]
                if len(t) > 0:
                    temp = t[0]
                else:
                    temp = 0

                ant_next_row_idx = ant_search_range[temp, 0]
                ant_next_col_idx = ant_search_range[temp, 1]

                if ant_next_row_idx is None:  # not sure about this..
                    ant_next_row_idx = ant_current_row_idx
                    ant_next_col_idx = ant_current_col_idx

                ant_pos_idx[ant_idx, 0] = ant_next_row_idx
                ant_pos_idx[ant_idx, 1] = ant_next_col_idx

                # record the delta_p_current
                delta_p_current[ant_pos_idx[ant_idx, 0], ant_pos_idx[ant_idx, 1]] = 1

                # record the new position into ant's memory
                if step_idx <= memory_length:
                    ant_memory[ant_idx, step_idx] = (ant_pos_idx[ant_idx, 0] - 1) * ncol + ant_pos_idx[ant_idx, 1]
                elif step_idx > memory_length:
                    ant_memory[ant_idx, :] = np.roll(ant_memory[ant_idx, :], -1, axis=0)
                    ant_memory[ant_idx, -1] = (ant_pos_idx[ant_idx, 0] - 1) * ncol + ant_pos_idx[ant_idx, 1]

                # update the pheromone function (10) in IEEE-CIM-06
                p = np.multiply((1 - rho), p)\
                    + np.multiply(np.multiply(np.multiply(rho, delta_p_current), v), delta_p_current)\
                    + np.multiply(p, abs(1 - delta_p_current))

            # update the pheromone function see equation (9) in IEEE-CIM-06

            delta_p = (delta_p + (delta_p_current > 0)) > 0

            p = np.multiply((1-phi), p)  # equation (9) in IEEE-CIM-06

print "FINISHED ", datetime.datetime.now().time()

