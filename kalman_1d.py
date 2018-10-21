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
import matplotlib.pyplot as mat

def kalman_gain(estimate_error, measurement_error):
    return estimate_error/(estimate_error + measurement_error)

def new_est(estimate, measurement, estimate_error, measurement_error, kg):
    return estimate + kg*(measurement - estimate)

def new_error(estimate_error, measurement_error, kg):
    return (1-kg)*estimate_error

def kalman_filter():
    #initialize all parameters
    estimate = 27.0
    estimate_error = 1.5
    iterations = input("Enter the number of iterations: ")

    true_value = 20
    measurement_error = 0.5
    measurements = np.random.normal(true_value, measurement_error, iterations) #draw measurements from a gaussian distribution with some error

    #keep track of estimates and error
    all_estimates = np.zeros(iterations, dtype=float)
    all_est_error = np.zeros(iterations, dtype=float)

    print(measurements)

    for i in range(iterations):
        #Kalman gain is calculated as part of new estimate and new error calculations
        kg = kalman_gain(estimate_error, measurement_error)
        estimate = new_est(estimate, measurements[i], estimate_error, measurement_error, kg)
        estimate_error = new_error(estimate_error, measurement_error, kg)

        #update estimates and error for later plotting
        all_estimates[i] = estimate
        all_est_error[i] = estimate_error

        print("Estimate: ", estimate, "\tMeasurement: ", measurements[i])
        print("Estimate error: ", estimate_error)
        print(kg)

    #get ready to plot
    x_axis = range(iterations)
    true_value_plot = [true_value for i in range(iterations)]
    
    fig, axis = mat.subplots(2, 1)

    axis[0].plot(x_axis, all_estimates, label="estimates")
    axis[0].plot(x_axis, measurements, label="measurements")
    axis[0].plot(x_axis, true_value_plot, label="true value")
    axis[0].set_ylabel("Temperature")
    axis[0].set_xlabel("Sample Number")
    axis[0].legend(loc="upper right", frameon=False)

    axis[1].plot(x_axis, all_est_error, label="error")
    axis[1].set_ylabel("Error")
    axis[1].set_xlabel("Sample Number")

    fig.tight_layout()
    mat.show()
        
kalman_filter();
