#include "RoveAdaptiveControl.h"

float sigmoid(const float z) {
    return 1.0 / (1.0 + exp(-z));
}
// // Assuming z are already results of sigmoid function,
// // if not it should be return Sigmoid(z) * (1.0 - Sigmoid(z));
// float sigmoid_derivative(const float z) {
//     return z * (1.0 - z);
// }

void RoveAdaptiveControl::PD_update(
    const Matrix<float, 1, INPUTS> &NN_In,
    const Matrix<float, OUTPUTS, 1> &r,  
    const Matrix<float, NEURONS - 1, 1> &first_layer_out)
{ 
    Matrix<float, NEURONS, 1> first_layer_out_bias;
    first_layer_out_bias << 1, first_layer_out;

    /* Calculate torque values for each joint */
    Matrix<float, INPUTS + NEURONS, NEURONS - 1 + OUTPUTS> norm_temp = Matrix<float, INPUTS + NEURONS, NEURONS - 1 + OUTPUTS>::Zero();
    norm_temp.topLeftCorner(INPUTS, NEURONS - 1)  = in_weights;
    norm_temp.bottomRightCorner(NEURONS, OUTPUTS) = out_weights;

    Matrix<float, OUTPUTS, 1> tau = K_v*r + out_weights.transpose()*first_layer_out_bias + K_z*(norm_temp.norm() - Z_B)*r;

    /* Calculate Decipercents from torques and conform to desired max value */
    int16_t J1_dec = max( -max_decipercent, min( max_decipercent, 1000*(tau[0]/max_torque_J1) ) );
    int16_t J2_dec = max( -max_decipercent, min( max_decipercent, 1000*(tau[1]/max_torque_J2) ) );
    int16_t J3_dec = max( -max_decipercent, min( max_decipercent, 1000*(tau[2]/max_torque_J3) ) );

    J1->drive(J1_dec);
    J2->drive(J2_dec);
    J3->drive(J3_dec);
}


void RoveAdaptiveControl::weight_update(
    const Matrix<float, 1, INPUTS> &NN_In,
    const Matrix<float, OUTPUTS, 1> &r,  
    const Matrix<float, NEURONS - 1, 1> &first_layer_out)
{    
    Matrix<float, NEURONS, 1> first_layer_out_bias;
    first_layer_out_bias << 1, first_layer_out;

    /* Create weight updates */
    Matrix<float, NEURONS, NEURONS> fl_bias_diag = first_layer_out_bias.asDiagonal();
    Matrix<float, INPUTS, NEURONS> in_weights_bias;
    in_weights_bias << Matrix<float, INPUTS, 1>::Ones(), in_weights;
    Matrix<float, NEURONS -1, NEURONS -1> fl_diag = first_layer_out.asDiagonal();

    Matrix<float, INPUTS, NEURONS - 1> in_weights_update = 
        (G_c*NN_In.transpose()*((fl_diag*(Matrix<float, NEURONS -1 , NEURONS -1>::Identity() - fl_diag))*out_weights(seq(1, last), all)*r).transpose()
        - kappa*G_c*r.norm()*in_weights)*update_t_delta;

    Matrix<float, NEURONS, OUTPUTS> out_weights_update = 
        (F_c*first_layer_out_bias*r.transpose() 
        - F_c*(fl_bias_diag*(Matrix<float, NEURONS, NEURONS>::Identity() - fl_bias_diag))*in_weights_bias.transpose()*NN_In.transpose()*r.transpose()
        - kappa*F_c*r.norm()*out_weights)*update_t_delta;

    in_weights += in_weights_update;
    out_weights += out_weights_update;
}


void RoveAdaptiveControl::calc_first_layer(
    Matrix<float, 1, INPUTS> &NN_In, 
    Matrix<float, OUTPUTS, 1> &r, 
    Matrix<float, NEURONS - 1, 1> &first_layer_out)
{
    Matrix<float, OUTPUTS, 1> pos;
    pos << 
        J1->m_encoder->readDegrees(), 
        J2->m_encoder->readDegrees(), 
        J3->m_encoder->readDegrees();

    Matrix<float, OUTPUTS, 1> vel; 
    vel << 
        J1->m_encoder->readVelocity(), 
        J2->m_encoder->readVelocity(), 
        J3->m_encoder->readVelocity();

    /* Update Errors */
    Matrix<float, OUTPUTS, 1> pos_error =  target_deg - pos;
    Matrix<float, OUTPUTS, 1> velo_error = target_deg_velo - vel;
    r = lam*pos_error + velo_error;

    /* Create Neural Net input vector */
    NN_In << 1, pos_error.transpose(), velo_error.transpose(), target_deg.transpose(), target_deg_velo.transpose(); // Could add acceleration

    /* Calulate hidden layer output */
    first_layer_out = in_weights.transpose() * NN_In.transpose();
    first_layer_out = first_layer_out.unaryExpr(std::ref(sigmoid));

    return;
}

void RoveAdaptiveControl::update_adaptive_control()
{
    Matrix<float, 1, INPUTS> NN_In;
    Matrix<float, OUTPUTS, 1> r;
    Matrix<float, NEURONS - 1, 1> first_layer_out;

    calc_first_layer(NN_In, r, first_layer_out);
    
    PD_update(NN_In, r, first_layer_out);

    weight_update(NN_In, r, first_layer_out);
}

void RoveAdaptiveControl::attach_joint(RoveJoint* joint, int joint_number)
{
    switch (joint_number)
    {
    case 1:
        J1 = joint;
        break;

    case 2:
        J2 = joint;
        break;

    case 3:
        J3 = joint;
        break;
    
    default:
        break;
    }
}

void RoveAdaptiveControl::set_target_angle_vel(float J1_ang, float J2_ang, float J3_ang, 
        float J1_vel /*= J1_default_vel*/, float J2_vel /*= J2_default_vel*/, float J3_vel /*= J3_default_vel*/)
{
    target_deg << J1_ang, J2_ang, J3_ang;
    target_deg_velo << J1_vel, J2_vel, J3_vel;
}

