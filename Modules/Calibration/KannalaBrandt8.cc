/**
* This file is part of Mini-SLAM
*
* Copyright (C) 2021 Juan J. Gómez Rodríguez and Juan D. Tardós, University of Zaragoza.
*
* Mini-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Mini-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with Mini-SLAM.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "KannalaBrandt8.h"

#define fx vParameters_[0]
#define fy vParameters_[1]
#define cx vParameters_[2]
#define cy vParameters_[3]
#define k0 vParameters_[4]
#define k1 vParameters_[5]
#define k2 vParameters_[6]
#define k3 vParameters_[7]

void KannalaBrandt8::project(const Eigen::Vector3f& p3D, Eigen::Vector2f& p2D){
    /*
     * Your code for Lab 3 - Task 5 here!
     */
    // Step 1: compute r = sqrt(x^2 + y^2) and theta = atan2(r, z)
    const float x = p3D[0];
    const float y = p3D[1];
    const float z = p3D[2];
 
    const float r = sqrtf(x * x + y * y);
    const float theta = atan2f(r, z);
 
    // Step 2: compute distorted radius d(theta) = theta + k0*theta^3 + k1*theta^5 + k2*theta^7 + k3*theta^9
    const float theta2 = theta * theta;
    const float theta3 = theta2 * theta;
    const float theta5 = theta3 * theta2;
    const float theta7 = theta5 * theta2;
    const float theta9 = theta7 * theta2;
 
    const float d = theta + k0 * theta3 + k1 * theta5 + k2 * theta7 + k3 * theta9;
 
    // Step 3: project to pixel coordinates
    // Guard against r = 0 (point on optical axis)
    if (r < 1e-8f) {
        p2D[0] = cx;
        p2D[1] = cy;
        return;
    }
 
    p2D[0] = fx * d * (x / r) + cx;
    p2D[1] = fy * d * (y / r) + cy;
}

void KannalaBrandt8::unproject(const Eigen::Vector2f& p2D, Eigen::Vector3f& p3D) {
    /*
     * Your code for Lab 3 - Task 5 here!
     */

    // Step 1: normalize pixel to undistorted coordinates
    const float mx = (p2D[0] - cx) / fx;
    const float my = (p2D[1] - cy) / fy;
 
    // Step 2: r' = sqrt(mx^2 + my^2)
    const float rp = sqrtf(mx * mx + my * my);
 
    // Step 3: recover theta from r' using Newton's method
    // We need to solve: d(theta) = rp
    // d(theta) = theta + k0*theta^3 + k1*theta^5 + k2*theta^7 + k3*theta^9
    float theta = rp; // initial guess
    for (int i = 0; i < 10; i++) {
        const float theta2 = theta * theta;
        const float theta3 = theta2 * theta;
        const float theta5 = theta3 * theta2;
        const float theta7 = theta5 * theta2;
        const float theta9 = theta7 * theta2;
 
        const float f   = theta + k0 * theta3 + k1 * theta5 + k2 * theta7 + k3 * theta9 - rp;
        const float fp  = 1.0f + 3.0f * k0 * theta2
                               + 5.0f * k1 * theta2 * theta2
                               + 7.0f * k2 * theta2 * theta2 * theta2
                               + 9.0f * k3 * theta2 * theta2 * theta2 * theta2;
 
        const float delta = f / fp;
        theta -= delta;
        if (fabsf(delta) < 1e-8f) break;
    }
 
    // Step 4: reconstruct direction vector
    // d = (sin(theta)*mx/r', sin(theta)*my/r', cos(theta))
    if (rp < 1e-8f) {
        p3D = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        return;
    }
 
    const float sinTheta = sinf(theta);
    const float cosTheta = cosf(theta);
 
    p3D[0] = sinTheta * (mx / rp);
    p3D[1] = sinTheta * (my / rp);
    p3D[2] = cosTheta;
}

void KannalaBrandt8::projectJac(const Eigen::Vector3f& p3D, Eigen::Matrix<float,2,3>& Jac) {
    /*
     * Your code for Lab 3 - Task 5 here!
     */
    // Inputs
    const float x = p3D[0];
    const float y = p3D[1];
    const float z = p3D[2];
 
    const float r2 = x * x + y * y;
    const float r  = sqrtf(r2);
 
    // Guard against singularity on optical axis
    if (r < 1e-8f) {
        Jac.setZero();
        return;
    }
 
    const float theta  = atan2f(r, z);
    const float theta2 = theta * theta;
    const float theta3 = theta2 * theta;
    const float theta4 = theta2 * theta2;
    const float theta5 = theta4 * theta;
    const float theta6 = theta4 * theta2;
    const float theta7 = theta6 * theta;
    const float theta8 = theta6 * theta2;
    const float theta9 = theta8 * theta;
 
    // d(theta) and its derivative w.r.t. theta
    const float d    = theta  + k0 * theta3 + k1 * theta5 + k2 * theta7 + k3 * theta9;
    const float dDot = 1.0f   + 3.0f * k0 * theta2
                              + 5.0f * k1 * theta4
                              + 7.0f * k2 * theta6
                              + 9.0f * k3 * theta8;
 
    // Partial derivatives of theta w.r.t. x, y, z
    // theta = atan2(r, z),  r = sqrt(x^2 + y^2)
    // d(theta)/dx = x*z / (r * (r^2 + z^2))
    // d(theta)/dy = y*z / (r * (r^2 + z^2))
    // d(theta)/dz = -r   / (r^2 + z^2)
    const float rz2  = r2 + z * z;   // = ||p3D||^2
    const float dTdx = x * z / (r * rz2);
    const float dTdy = y * z / (r * rz2);
    const float dTdz = -r / rz2;
 
    // Partial derivatives of (x/r) and (y/r) w.r.t. x, y, z
    // d(x/r)/dx = y^2 / r^3
    // d(x/r)/dy = -x*y / r^3
    // d(x/r)/dz = 0
    const float r3 = r2 * r;
    const float dxr_dx =  y * y / r3;
    const float dxr_dy = -x * y / r3;
    const float dyr_dx = -x * y / r3;
    const float dyr_dy =  x * x / r3;
 
    // u = fx * d(theta) * (x/r) + cx
    // du/dx = fx * (dDot * dTdx * (x/r)  +  d * dxr_dx)
    // du/dy = fx * (dDot * dTdy * (x/r)  +  d * dxr_dy)
    // du/dz = fx * (dDot * dTdz * (x/r))
    const float xr = x / r;
    const float yr = y / r;
 
    Jac(0, 0) = fx * (dDot * dTdx * xr + d * dxr_dx);
    Jac(0, 1) = fx * (dDot * dTdy * xr + d * dxr_dy);
    Jac(0, 2) = fx *  dDot * dTdz * xr;
 
    Jac(1, 0) = fy * (dDot * dTdx * yr + d * dyr_dx);
    Jac(1, 1) = fy * (dDot * dTdy * yr + d * dyr_dy);
    Jac(1, 2) = fy *  dDot * dTdz * yr;
}

void KannalaBrandt8::unprojectJac(const Eigen::Vector2f& p2D, Eigen::Matrix<float,3,2>& Jac) {
    throw std::runtime_error("KannalaBrandt8::unprojectJac not implemented yet");
}