#  Filename: kalman_1d.py
#  Author: Zach Sherer
#  Purpose: Implements a 1d kalman filter on scalar values for a single input parameter.
#  Notes:
#    some definitions:
#     gain: kalman gain vector for all measurements
#     estimate_error: error in all estimates. Consider this the process noise.
#     estimate: all estimates. holds the current estimate outside of functions, but will act as 
#                   the previous estimate when used as a function arg. This will make sense in the code.
#     measurement_error: error in all measurements. Consider this the measurement noise.
#     measurement: holds all measurements. Has the same properties as estimate.

import numpy as np
import matplotlib as mp

def kalman_gain(estimate_error, measurement_error):
    return estimate_error/(estimate_error * measurement_error)

def new_est(estimate, measurement, estimate_error, measurement_error):
    kg = kalman_gain(estimate_error, measurement_error)
    return estimate + kg*(measurement - estimate)

def new_error(estimate_error, measurement_error):
    kg = kalman_gain(estimate_error, measurement_error)
    return (1-kg)*estimate_error

def kalman_filter():
    #initialize all parameters
    estimate = 27.0
    estimate_error = 1.5

    true_value = 20
    measurement_error = 0.5
    measurements = np.random.normal(true_value, measurement_error, 100) #draw measurements from a gaussian distribution with some error

    #keep track of estimates and error
    all_estimates = np.zeros(100, dtype=float)
    all_est_error = np.zeros(100, dtype=float)

    for i in range(100):
        #Kalman gain is calculated as part of new estimate and new error calculations
        kg = kalman_gain(estimate_error, measurement_error)
        estimate = new_est(estimate, measurements[i], estimate_error, measurement_error)
        estimate_error = new_error(estimate_error, measurement_error)

        #update estimates and error for later plotting
        all_estimates[i] = estimate
        all_est_error[i] = estimate_error

        print("Estimate: ", estimate, "\tMeasurement: ", measurements[i])
        print("Estimate error: ", estimate_error)
        print("Kalman gain: ", kg)
        
kalman_filter();
