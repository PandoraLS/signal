close all;clear all;clc;

%% ���������ź�
fs = 1;
f0 = 0.02;
n = 1000;
t = (0:n-1)'/fs;
xs = cos(2*pi*f0*t);
ws = awgn(xs, 20, 'measured'); % ��Ӱ�������20dB

% figure;
% subplot(211);plot(t, xs);title('ԭʼ�ź�');
% subplot(212);plot(t, ws);title('�����ź�');

M  = 20 ;   % �˲����Ľ���
xn = ws;    % �����ź�
dn = xs;    % ԭʼ�ź�(�����ź�)
% rho_max = max(eig(ws*ws.'));   % �����ź���ؾ�����������ֵ
% mu = (1/rho_max) ;    % �������� 0 < mu < 1/rho
mu = 0.001;
[yn,W,en] = lmsFunc(xn,dn,M,mu);

figure;
ax1 = subplot(211);
plot(t,ws);grid on;ylabel('��ֵ');xlabel('ʱ��');
ylim([-1.5 1.5]);title('LMS�˲��������ź�');

ax2 = subplot(212);
plot(t,yn);grid on;ylabel('��ֵ');xlabel('ʱ��');title('LMS�˲�������ź�');
ylim([-1.5 1.5]);linkaxes([ax1, ax2],'xy');

figure;plot(en);grid on;title('���');