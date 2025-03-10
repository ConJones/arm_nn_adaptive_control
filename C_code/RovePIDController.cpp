#include "RovePIDController.h"

#include <cmath>


void RovePIDController::configPID(const float& kP, const float& kI, const float& kD) {
    configKP(kP);
    configKI(kI);
    configKD(kD);
}

void RovePIDController::configKP(const float& kP) {
    m_kP = kP;
}

void RovePIDController::configKI(const float& kI) {
    m_kI = kI;
}

void RovePIDController::configKD(const float& kD) {
    m_kD = kD;
}



void RovePIDController::configIZone(const float& iZone) {
    m_iZone = iZone;
}

void RovePIDController::configMaxIntegralAccum(const float& max) {
    m_maxIntegralAccum = max;
}



void RovePIDController::configOutputLimits(const float& max, const float& min) {
    configMaxOutput(max);
    configMinOutput(min);
}

void RovePIDController::configMaxOutput(const float& max) {
    m_maxOutput = max;
}

void RovePIDController::configMinOutput(const float& min) {
    m_minOutput = min;
}



void RovePIDController::reset() {
    m_firstLoop = true;
}

float RovePIDController::calculate(const float& target, const float& feedback, const float& timestamp) {
    float error = target - feedback;
    float derivative;

    if (m_firstLoop) {
        derivative = 0;
        m_integral = 0;
        m_firstLoop = false;
    }
    else {
        float dt = timestamp - m_lastTimestamp;
        if (dt == 0) derivative = 0;
        else derivative = (error - m_lastError) / dt;

        if (std::abs(error) > m_iZone) {
            m_integral = 0;
        }
        else {
            m_integral += error * dt;
            if (m_integral > m_maxIntegralAccum) m_integral = m_maxIntegralAccum;
            else if (m_integral < -m_maxIntegralAccum) m_integral = -m_maxIntegralAccum;
        }
    }
    
    m_lastTimestamp = timestamp;
    m_lastError = error;

    float output = m_kP*error + m_kI*m_integral + m_kD*derivative;
    
    if(output > m_maxOutput) return m_maxOutput;
    if(output < m_minOutput) return m_minOutput;
    return output;
}