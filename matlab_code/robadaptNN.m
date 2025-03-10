function xdot1=robadaptNN(t,x)
%--------------------------------------------------------------------------
% NN control with Backprop weight tuning
%--------------------------------------------------------------------------
global angles
global angles_t
global I
global Outs
global NN_delta_t
global PD_delta_t
global tau
global PD_step_num
global NN_step_num
persistent W
if isempty(W)
   W=zeros(I,Outs);
end
%Compute desired trajectory
angles_idx=round(t/angles_t(2))+1;
qd=angles(angles_idx,1:3)';
if(angles_idx > 1)
   qdp=((angles(angles_idx,1:3)-angles(angles_idx-1,1:3))/angles_t(2))';
   if(angles_idx > 2)
       qdpp=(((angles(angles_idx,1:3)-angles(angles_idx-1,1:3)) - (angles(angles_idx-1,1:3)-angles(angles_idx-2,1:3)))/angles_t(2)^2)';
   else
       qdpp=zeros(Outs,1);
   end
else
   qdp=zeros(Outs,1);
   qdpp=zeros(Outs,1);
end
%Neural Net
q=[x(1) x(2) x(3)]';
qp=[x(4) x(5) x(6)]';
if( t > PD_delta_t*PD_step_num)
   %Adaptive control input
   Kv=20*eye(Outs); lam=5*eye(Outs); F=50*eye(I); % controller parameters

   %tracking errors
   e=qd-q;
   ep=qdp-qp;
   r=ep+lam*e;

   %compute regression matrix
   %f=qdpp+lam*ep;
   NNIn(1)=1;NNIn(2:2+Outs-1)=e';NNIn(2+Outs:2+2*Outs-1)=ep';
   NNIn(2+2*Outs:2+3*Outs-1)=qd';NNIn(2+3*Outs:2+4*Outs-1)=qdp';NNIn(2+4*Outs:2+5*Outs-1)=qdpp';
   %XX=V'*NNIn';

   NNOut=logsig(NNIn');
   NNOut(1) = 1;

   %control torques. Parameter estimates are [x(5) x(6)]'
   tau=Kv*r+W'*NNOut;
   PD_step_num = PD_step_num +1;
end
if( t > NN_delta_t*NN_step_num )
   %parameter updates
   T_w_dot=(F*NNOut*r')*NN_delta_t;
   NN_step_num = NN_step_num +1;
else
   T_w_dot=zeros(size(W));
end
%parameter updates
W = W + T_w_dot;
% Robot Arm Dynamics
% m1 = weight of first link, m2 = weight of second link, m3 = payload
% weight, a1 = length of first link, a2 = length of second link
m1=0.8; m2=2.3; m3=2.7; a1=1; a2=1; g=9.8; % arm parameters
if t > 14
m3 = 2.7+1.5;
end
%Inertia M(q)
M11=(m3+m2+0.25*m1)*a1^2*(cos(q(2)))^2 + (m3+0.25*m2)*a2^2*(sin(q(2)+q(3)))^2 ...
    + (m3+m2)*a1*a2*cos(q(2))*sin(q(2)+q(3)) + 1;
M22=(m3+m2+0.25*m1)*a1^2 + (m3+0.25*m2)*a2^2 + (2*m3+m2)*a1*a2*sin(q(3)) + 1;
M23=(m3+0.25*m2)*a2^2 + (m3 + 0.5*m2)*a1*a2*sin(q(3)) + 1;
M32=M23;
M33=(m3+0.25*m2)*a2^2 + 1;
M=[M11 0 0; 0 M22 M23; 0 M32 M33];
% Coriolis/centripetal vector V(q,qdot)
V1=qp(1)*qp(2)*((m3+0.25*m2)*a2^2*sin(2*(q(2)+q(3)))) - (m3+m2+0.25*m1)*a1^3*sin(2*q(2)) ...
   + (m3+m2)*a1*a2*cos(2*q(2)+q(3)) + qp(1)*qp(3)*((m3+0.25*m2)*a2^2*sin(2*(q(2)+q(3)))) ...
   + (m3+m2)*a1*a2*cos(q(2))*cos(q(2)+q(3));
V2=qp(2)*qp(3)*(-(2*m3+m2)*a1*a2*cos(q(3))) + q(3)^2*(-(m3+0.5*m2)*a1*a2*cos(q(3))) ...
   + 0.5*qp(1)^2*((m3+m2+0.25*m1)*a1^2*sin(2*q(2)) + (m3+0.25*m2)*a2^2*sin(2*(q(2)+q(3))) ...
   + (m3+0.5*m2)*a1*a2*cos(2*q(2)+q(3)));
V3=-0.5*qp(1)^2*((m3+0.25*m2)*a2^2*sin(2*(q(2)+q(3))) + (m3 + m2)*a1*a2*cos(q(2))*cos(q(2)+q(3))) ...
   - (0.5*qp(2)^2+qp(2)*qp(3))*(2*m3+m2)*a1*a2*cos(q(3));
V = [V1 V2 V3]';
% Gravity vector G(q)
G1=0;
G2=g*((m3+m2+0.5*m1)*a1*cos(q(2)) + (m3+0.5*m2)*a2*sin(q(2)+q(3)));
G3=g*(m3+0.5*m2)*a2*sin(q(2)+q(3));
G = [G1 G2 G3]';
% Manipulator Dynamics
qpp = M\(tau - V - G);
%state equations
xdot(1)=x(4);
xdot(2)=x(5);
xdot(3)=x(6);
xdot(4)=qpp(1);
xdot(5)=qpp(2);
xdot(6)=qpp(3);
xdot1=xdot';
endfunction

