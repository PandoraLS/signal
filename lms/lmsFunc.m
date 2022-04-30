% 输入参数:
%   xn   输入的信号
%   dn   所期望的响应
%   M    滤波器的阶数
%   mu   收敛因子(步长)
% 输出参数:
%   W    滤波器系数矩阵  
%   en   误差序列 
%   yn   滤波器输出        
function [yn, W, en]=lmsFunc(xn, dn, M, mu)
itr = length(xn);
en = zeros(itr,1);            
W  = zeros(M,itr);    % 每一列代表-次迭代,初始为0
% 迭代计算
for k = M:itr                  % 第k次迭代
    x = xn(k:-1:k-M+1);        % 滤波器M个抽头的输入
    y = W(:,k-1)' * x;        % 滤波器的输出
    en(k) = dn(k) - y ;        % 第k次迭代的误差
    % 滤波器权值计算的迭代式
    W(:,k) = W(:,k-1) + 2*mu*en(k)*x;
end

yn = inf * ones(size(xn)); % 初值为无穷大是绘图使用，无穷大处不会绘图
for k = M:length(xn)
    x = xn(k:-1:k-M+1);
    yn(k) = W(:,end)'* x;  % 最终输出结果
end