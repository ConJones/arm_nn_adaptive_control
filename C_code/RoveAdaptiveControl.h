#ifndef ROVE_ADAPTIVE_CONTROL_H
#define ROVE_ADAPTIVE_CONTROL_H

#include <TimerOne.h>
#include <RoveJoint.h>
#include <ArduinoEigenDense.h>
#include <math.h>
using namespace Eigen;

#define INPUTS (1 + 4*OUTPUTS)
#define OUTPUTS 3
#define NEURONS (10 + 1) // Extra neuron to represent the bias

#define K_z (50.0f * Matrix<float, OUTPUTS, OUTPUTS>::Identity())
#define K_v (20.0f * Matrix<float, OUTPUTS, OUTPUTS>::Identity())
#define lam (5.0f  * Matrix<float, OUTPUTS, OUTPUTS>::Identity())
#define F_c (50.0f * Matrix<float, NEURONS, NEURONS>::Identity())
#define G_c (50.0f * Matrix<float, INPUTS, INPUTS>::Identity())
#define kappa 0.1f
#define Z_B 0.1f
#define update_t_delta 0.002f
#define max_torque_J1 15
#define max_torque_J2 150
#define max_torque_J3 70
#define max_decipercent 800

#define J1_default_vel 30.0f
#define J2_default_vel 30.0f
#define J3_default_vel 30.0f


class RoveJoint;

class RoveAdaptiveControl
{
private:
    /* data */

    RoveJoint* J1 = nullptr;
    RoveJoint* J2 = nullptr;
    RoveJoint* J3 = nullptr;

    /* In matlab as V */
    Matrix<float, INPUTS, NEURONS - 1> in_weights;
    /* In matlab as W */
    Matrix<float, NEURONS, OUTPUTS> out_weights;

    /* In matlab as qd */
    Matrix<float, OUTPUTS, 1> target_deg;
    /* In matlab as qdp */
    Matrix<float, OUTPUTS, 1> target_deg_velo;

    /**
	 * @brief Calculates the output of the first layer and necessary inputs
	 * 
	 * @param NN_in Empty container to be filled with neural net inputs
     * @param r Empty container to be filled with the filtered tracking error
     * @param first_layer_out Empty container to be filled with the output of the hidden layer neurons
	 */
    void calc_first_layer(
        Matrix<float, 1, INPUTS> &NN_In, 
        Matrix<float, OUTPUTS, 1> &r, 
        Matrix<float, NEURONS - 1, 1> &first_layer_out);

public:

    /**
	 * @brief Calculates the torques for each joint and drives motors
	 * 
	 * @param NN_in Container filled with neural net inputs
     * @param r Container filled with the filtered tracking error
     * @param first_layer_out Container filled with the output of the hidden layer neurons
	 */
    void PD_update(
        const Matrix<float, 1, INPUTS> &NN_In,
        const Matrix<float, OUTPUTS, 1> &r,  
        const Matrix<float, NEURONS - 1, 1> &first_layer_out);

    /**
	 * @brief Updates neural network weights
	 * 
	 * @param NN_in Container filled with neural net inputs
     * @param r Container filled with the filtered tracking error
     * @param first_layer_out Container filled with the output of the hidden layer neurons
	 */
    void weight_update(
        const Matrix<float, 1, INPUTS> &NN_In,
        const Matrix<float, OUTPUTS, 1> &r,  
        const Matrix<float, NEURONS - 1, 1> &first_layer_out);

    /**
	 * @brief Attaches a joint to the adaptive control structure
	 * 
	 * @param joint Pointer to the joint object to be attached
     * @param joint_number The joint number to be attached
	 */
    void attach_joint(RoveJoint* joint, int joint_number);

    /**
	 * @brief Main adaptive control function, updates weights and drives motors
	 */
    void update_adaptive_control();

    /**
	 * @brief Main adaptive control function, updates weights and drives motors
	 */
    void set_target_angle_vel(float J1_ang, float J2_ang, float J3_ang, 
        float J1_vel = J1_default_vel, float J2_vel = J2_default_vel, float J3_vel = J3_default_vel);
};


#endif