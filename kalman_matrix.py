#  Filename: kalman_matrix.py
#  Author: Zach Sherer
#  Purpose: Implements a 2d Kalman filter to track an object's position and velocity in one direction.
#  Notes: Process is much the same as that in kalman_1d.py

import numpy as np
import matplotlib.pyplot as plt

measurement_error = 0.1

def genMeasurement(true_position, true_velocity, C_matrix):
    measurement = []
    #create slightly noisy measurements
    measurement.append(np.random.normal(true_position, measurement_error))
    measurement.append(np.random.normal(true_velocity, measurement_error))

    return np.matmul(C_matrix, measurement)

def genProcessCovariance(process_covariance, A_matrix):
    a_transpose = np.matrix.transpose(np.asarray(A_matrix))
    intermediate_product = np.matmul(A_matrix, process_covariance)
    return np.matmul(a_transpose, intermediate_product)

def genPredictedState(state, control_matrix, A_matrix, B_matrix): #control matrix has stochastic accelerations in x direction
    predicted_state = np.matmul(A_matrix, state) + np.array(B_matrix)*control_matrix
    return predicted_state
    
def kalmanGain(process_covariance, R_matrix):
    h_matrix = np.identity(2)
    numerator = np.matmul(process_covariance, np.matrix.transpose(h_matrix))
    denominator = np.matmul(np.matrix.transpose(h_matrix), np.matmul(h_matrix, process_covariance)) + R_matrix
    return np.matmul(numerator, np.linalg.inv(denominator))

def genNextState(gain, predicted_state, measurement):
    h_matrix = np.identity(2)
    difference = measurement - predicted_state
    gaindiff = np.matmul(gain, difference)
    return predicted_state + gaindiff

def updateProcessCovariance(gain, process_covariance):
    h_matrix = np.identity(2)
    gain_transform = h_matrix - np.matmul(gain, h_matrix)
    return np.matmul(gain_transform, process_covariance)

def matrixKalman():
    #define some constants
    T = 0.5 #delta T for sampling
    A_matrix = [[1, T], [0, 1]]
    B_matrix = [0.5*(T**2), T]
    A_matrix = np.array(A_matrix)
    B_matrix = np.array(B_matrix)

    C_matrix = np.identity(2)

    iterations = input("Enter the number of iterations: ")

    all_estimates = []
    all_est_error = []
    all_measurements = []

    estimate = [20, 3]

    #error matrices
    error_pos_proc = 0.5
    error_vel_proc = 0.5
    error_pos_obs  = 0.75
    error_vel_obs  = 0.75

    process_covariance = np.array([[error_pos_proc**2, 0], [0, error_vel_proc**2]]) #covariances set to 0 due to lack of correlation in the measurements
    R_matrix = np.array([[error_pos_obs**2, 0], [0, error_vel_proc**2]])
    print("Initial process covariance:")
    print(process_covariance)
    print(R_matrix)

    all_estimates.append(estimate[:])
    all_est_error.append(process_covariance[:])

    #generate predictable values for the true value, maybe a line
    true_positions = range(iterations);
    true_velocity = 2

    for i in range(iterations):
        print("Round ", i)
        print("\n")

        control_matrix = np.random.normal(0, 0.5) #random acceleration
        predicted_state = genPredictedState(estimate, control_matrix, A_matrix, B_matrix)
        print("predicted_state:")
        print(predicted_state)
        process_covariance = genProcessCovariance(process_covariance, A_matrix)
        print("Predicted process covariance:")
        print(process_covariance)
        kg = kalmanGain(process_covariance, R_matrix)
        print("gain:")
        print(kg)
        measurement = genMeasurement(true_positions[i], true_velocity, C_matrix)
        print("measurement:")
        print(measurement)
        estimate = genNextState(kg, predicted_state, measurement)
        print("next_state:")
        print(estimate)
        process_covariance = updateProcessCovariance(kg, process_covariance)
        print("process covariance:")
        print(process_covariance)

        all_estimates.append(estimate[:])
        all_est_error.append(process_covariance[:])
        all_measurements.append(measurement)
        
    true_velocities = [true_velocity for i in range(iterations)]
    pred_positions = []
    pred_velocities = []
    meas_positions = []
    meas_velocities = []
    x_values = range(iterations)
    for i in range(iterations):
        pred_pos = all_estimates[i][0]
        pred_vel = all_estimates[i][1]
        pred_positions.append(pred_pos)
        pred_velocities.append(pred_vel)
        meas_pos = all_measurements[i][0]
        meas_vel = all_measurements[i][1]
        meas_positions.append(meas_pos)
        meas_velocities.append(meas_vel)

    fig, axis = plt.subplots(2, 1)
    axis[0].plot(x_values, true_positions, label="true")
    axis[0].plot(x_values, meas_positions, label="measured")
    axis[0].plot(x_values, pred_positions, label="predicted")
    axis[0].set_xlabel("Sample Number")
    axis[0].set_ylabel("Position")
    axis[0].legend(loc="upper right", frameon=False)

    axis[1].plot(x_values, true_velocities, label="true")
    axis[1].plot(x_values, meas_velocities, label="measured")
    axis[1].plot(x_values, pred_velocities, label="predicted")
    axis[1].set_xlabel("Sample Number")
    axis[1].set_ylabel("Velocity")
    axis[1].legend(loc="upper right", frameon=False)
    

    plt.show()


matrixKalman()
