#ifndef ROVEPIDCONTROLLER_H
#define ROVEPIDCONTROLLER_H

#include <cfloat>

class RovePIDController {

private:

    float m_kP = 0, m_kI = 0, m_kD = 0;
    float m_iZone = FLT_MAX, m_maxIntegralAccum = FLT_MAX;
    float m_maxOutput = FLT_MAX, m_minOutput = -FLT_MAX;
    float m_lastError, m_lastTimestamp, m_integral;
    bool m_firstLoop = true;

public:

    /**
     * @brief Construct a new RovePIDController object.
     * 
     */
    RovePIDController() {}

    /**
     * @brief Construct a new RovePIDController object.
     * 
     * @param kP 
     * @param kI 
     * @param kD 
     */
    RovePIDController(const float& kP, const float& kI, const float& kD) : m_kP(kP), m_kI(kI), m_kD(kD) {}


    /**
     * @brief Configure the gains of the PID Controller.
     * 
     * @param kP 
     * @param kI 
     * @param kD 
     */
    void configPID(const float& kP, const float& kI, const float& kD);

    /**
     * @brief Configure the proportional gain of the PID controller.
     * 
     * @param kP 
     */
    void configKP(const float& kP);

    /**
     * @brief Configure the integral gain of the PID controller.
     * 
     * @param kI 
     */
    void configKI(const float& kI);

    /**
     * @brief Configure the derivative gain of the PID controller.
     * 
     * @param kD 
     */
    void configKD(const float& kD);


    /**
     * @brief Configure the integral zone of the PID controller. If the absolute error is greater than this value
     * when calculate() is called, the integral will be reset to 0.
     * 
     * @param iZone [0, FLT_MAX], defaults to FLT_MAX.
     */
    void configIZone(const float& iZone);

    /**
     * @brief Configure the maximum integral accumulation of the PID controller. If the integral is greater than this
     * value when calculate() is called, it will be set to equal this value.
     * 
     * @param max [0, FLT_MAX], defaults to FLT_MAX.
     */
    void configMaxIntegralAccum(const float& max);


    /**
     * @brief Configure the maximum and minimum values returned by calculate().
     * 
     * @param max 
     * @param min 
     */
    void configOutputLimits(const float& max, const float& min);

    /**
     * @brief Configure the maximum value returned by calculate().
     * 
     * @param max 
     */
    void configMaxOutput(const float& max);

    /**
     * @brief Configure the minimum value returned by calculate().
     * 
     * @param min 
     */
    void configMinOutput(const float& min);


    /**
     * @brief Reset the PID controller. When calculate() is next called, the derivative and integral terms will be 0.
     * 
     */
    void reset();

    /**
     * @brief Calculate the output of the PID controller.
     * 
     * @param target
     * @param feedback
     * @param timestamp 
     * @return The output of the PID controller, bounded by the configured maximum and minimum values. 
     */
    float calculate(const float& target, const float& feedback, const float& timestamp);

};

#endif