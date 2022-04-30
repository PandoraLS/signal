% �������:
%   xn   ������ź�
%   dn   ����������Ӧ
%   M    �˲����Ľ���
%   mu   ��������(����)
% �������:
%   W    �˲���ϵ������  
%   en   ������� 
%   yn   �˲������        
function [yn, W, en]=lmsFunc(xn, dn, M, mu)
itr = length(xn);
en = zeros(itr,1);            
W  = zeros(M,itr);    % ÿһ�д���-�ε���,��ʼΪ0
% ��������
for k = M:itr                  % ��k�ε���
    x = xn(k:-1:k-M+1);        % �˲���M����ͷ������
    y = W(:,k-1)' * x;        % �˲��������
    en(k) = dn(k) - y ;        % ��k�ε��������
    % �˲���Ȩֵ����ĵ���ʽ
    W(:,k) = W(:,k-1) + 2*mu*en(k)*x;
end

yn = inf * ones(size(xn)); % ��ֵΪ������ǻ�ͼʹ�ã�����󴦲����ͼ
for k = M:length(xn)
    x = xn(k:-1:k-M+1);
    yn(k) = W(:,end)'* x;  % ����������
end