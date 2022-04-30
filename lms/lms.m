close all;clear all;clc;

%% 产生测试信号
fs = 1;
f0 = 0.02;
n = 1000;
t = (0:n-1)'/fs;
xs = cos(2*pi*f0*t);
ws = awgn(xs, 20, 'measured'); % 添加白噪声，20dB

% figure;
% subplot(211);plot(t, xs);title('原始信号');
% subplot(212);plot(t, ws);title('加噪信号');

M  = 20 ;   % 滤波器的阶数
xn = ws;    % 加噪信号
dn = xs;    % 原始信号(期望信号)
% rho_max = max(eig(ws*ws.'));   % 输入信号相关矩阵的最大特征值
% mu = (1/rho_max) ;    % 收敛因子 0 < mu < 1/rho
mu = 0.001;
[yn,W,en] = lmsFunc(xn,dn,M,mu);

figure;
ax1 = subplot(211);
plot(t,ws);grid on;ylabel('幅值');xlabel('时间');
ylim([-1.5 1.5]);title('LMS滤波器输入信号');

ax2 = subplot(212);
plot(t,yn);grid on;ylabel('幅值');xlabel('时间');title('LMS滤波器输出信号');
ylim([-1.5 1.5]);linkaxes([ax1, ax2],'xy');

figure;plot(en);grid on;title('误差');