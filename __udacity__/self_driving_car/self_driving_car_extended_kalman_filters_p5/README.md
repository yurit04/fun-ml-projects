# **Extended Kalman Filter Project (C++)** 


In this Udacity's Self-Driving Car Engineer project we utilize a Kalman filter to estimate the state of a moving object of interest with noisy **lidar** and **radar** measurements. Passing the project requires obtaining RMSE values (i.e. RMSE of estimated vs ground truth states) that are lower than the required tolerance outlined in the project rubric. 

This project uses the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases).


The general Kalman Filter algorithm is comprised of the following steps:

* **first measurement** - the filter will receive initial measurements of the bicycle's position relative to the car. These measurements will come from a radar or lidar sensor.
* **initialize state and covariance matrices** - the filter will initialize the bicycle's position based on the first measurement.
* then the car will receive another sensor measurement after a time period Δt.
* **predict** - the algorithm will predict where the bicycle will be after time Δt. One basic way to predict the bicycle location after Δt is to assume the bicycle's velocity is constant; thus the bicycle will have moved velocity * Δt. In the extended Kalman filter lesson, we will assume the velocity is constant.
* **update** - the filter compares the "predicted" location with what the sensor measurement says. The predicted location and the measured location are combined to give an updated location. The Kalman filter will put more weight on either the predicted location or the measured location depending on the uncertainty of each value.
* then the car will receive another sensor measurement after a time period Δt. The algorithm then does another **predict** and **update** step.

Since the project involves processing inputs from two sensors (Lidar and Radar) we are implementing a **Sensor Fusion** process. Athough the laser/lidar provides the object positions with relatively high accuracy, there is no way to observe directly the object's speed. This is where radar comes in by providing radial velocity (along with radial distance and bearing angle). At the same time, radar has a lower resolution than the laser sensors. Hence it makes sense to combine both to obtain more accurate estimation of the object's state (position and velocity).
 
Additionally, given that radar measurements are in polar coordinates, one of the standard Kalman filter assumptions (linear measurement transformation, i.e. a matrix H) does not hold. Instead, the approach utilizes a first order approximation (i.e. Jacobian matrix of the measurement transform). Hence the approach is called **Extended Kalman Filter**. For maths behind the approach see ***sensor-fusion-ekf-reference.pdf***.


The project requires implementing the extended Kalman filter in C++. The simulator provides simulated lidar and radar measurements detecting a bicycle that travels around the vehicle. Extended Kalman filter is then used by processing lidar measurements and radar measurements to track the bicycle's position and velocity. 
For an example output see this [video](https://www.youtube.com/watch?v=d6qbR3_LPoA).
Lidar measurements are red circles, radar measurements are blue circles with an arrow pointing in the direction of the observed angle, and estimation markers are green triangles. The video shows what the simulator looks like when a c++ script is using its Kalman filter to track the object. The simulator provides the script the measured data (either lidar or radar), and the script feeds back the measured estimation marker, and RMSE values from its Kalman filter.


---
### Submitted Files & Code


The files we work with are in the src folder of the github repository.

* *main.cpp* - communicates with the Term 2 Simulator receiving data measurements, calls a function to run the Kalman filter, calls a function to calculate RMSE
* *FusionEKF.cpp* - initializes the filter, calls the predict function, calls the update function
* *kalman_filter.cpp* - defines the predict function, the update function for lidar, and the update function for radar
* *tools.cpp* - function to calculate RMSE and the Jacobian matrix

The files we need to modify are *FusionEKF.cpp*, *kalman_filter.cpp*, and *tools.cpp*.

Here is a brief overview of what happens when we run the code files:

* *Main.cpp* reads in the data and sends a sensor measurement to FusionEKF.cpp
* *FusionEKF.cpp* takes the sensor data and initializes variables and updates variables. The Kalman filter equations are not in this file. FusionEKF.cpp has a variable called ekf_, which is an instance of a KalmanFilter class. The ekf_ will hold the matrix and vector values. You will also use the ekf_ instance to call the predict and update equations.
* The KalmanFilter class is defined in *kalman_filter.cpp* and *kalman_filter.h*. We need to modify *kalman_filter.cpp*, which contains functions for the prediction and update steps.

