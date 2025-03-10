clear all;
clc;
global angles
global angles_t
global I
global L
global Outs
global NN_delta_t
global PD_delta_t
global tau;
global PD_step_num
global NN_step_num
NN_delta_t = 0.02;
PD_delta_t = NN_delta_t/80; %0.059/5 is max step size
tspan=[0, 20];
angles=load("joint_angles.mat");
angles=pi*angles.angle_export/180;
%angles = [angles ; angles];
angles_len=length(angles);
angles_t=0:((tspan(2)+(tspan(2)/angles_len))/angles_len):tspan(2);
I = 16; % Num of NN inputs
L=10; % Num hidden layer neurons
L=L+1;
Outs=3; % Num NN outputs
x0_PD = [0 0 0 0 0 0]';
tau=0;
PD_step_num=0;
NN_step_num=0;
[t_2LNN,x_2LNN]=ode45('robadapt2LNN',tspan,x0_PD);
tau=0;
PD_step_num=0;
NN_step_num=0;
[t_2LNNAT,x_2LNNAT]=ode45('robadapt2LNNAT',tspan,x0_PD);
tau=0;
PD_step_num=0;
NN_step_num=0;
[t_NN,x_NN]=ode45('robadaptNN',tspan,x0_PD);
tau=0;
PD_step_num=0;
NN_step_num=0;
[t_NNAT,x_NNAT]=ode45('robadaptNNAT',tspan,x0_PD);
tau=0;
PD_step_num=0;
NN_step_num=0;
[t_PD, x_PD]=ode45('robadaptPD', tspan,x0_PD);
figure(1)
plot(t_2LNN,x_2LNN(:,1:3),angles_t,angles(:,1:3),'-.');
title("Two Layer Neural Net");
angles_idx=round(t_2LNN/angles_t(2))+1;
error = angles(angles_idx,1:3) - x_2LNN(:,1:3);
mean_2LNN = mean(abs(error));
figure(2)
plot(t_2LNNAT,x_2LNNAT(:,1:3),angles_t,angles(:,1:3),'-.');
title("Two Layer Neural Net Augmented Tuning")
angles_idx=round(t_2LNNAT/angles_t(2))+1;
error = angles(angles_idx,1:3) - x_2LNNAT(:,1:3);
mean_2LNNAT = mean(abs(error));
figure(3)
plot(t_NN,x_NN(:,1:3),angles_t,angles(:,1:3),'-.');
title("One Layer Neural Net")
angles_idx=round(t_NN/angles_t(2))+1;
error = angles(angles_idx,1:3) - x_NN(:,1:3);
mean_NN = mean(abs(error));
figure(4)
plot(t_NNAT,x_NNAT(:,1:3),angles_t,angles(:,1:3),'-.');
title("One Layer Neural Net Augmented Tuning")
angles_idx=round(t_NNAT/angles_t(2))+1;
error = angles(angles_idx,1:3) - x_NNAT(:,1:3);
mean_NNAT = mean(abs(error));
figure(5)
plot(t_PD,x_PD(:,1:3),angles_t,angles(:,1:3),'-.');
title("Classical PD Controller")
angles_idx=round(t_PD/angles_t(2))+1;
error = angles(angles_idx,1:3) - x_PD(:,1:3);
mean_PD = mean(abs(error));
%figure(6)
%names = categorical(["2LNN" "2LNNAT" "1LNN" "1LNNAT" "PD"]);
%names = reordercats(names,["2LNN" "2LNNAT" "1LNN" "1LNNAT" "PD"]);
%values = [mean(mean_2LNN) mean(mean_2LNNAT) mean(mean_NN) mean(mean_NNAT) mean(mean_PD)];
%bar(names , values)

