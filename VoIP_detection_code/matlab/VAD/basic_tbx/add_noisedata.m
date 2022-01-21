function [signal,noise]=add_noisedata(s,data,fs,fs1,snr)
s=s(:);                          % ���ź�ת����������
s=s-mean(s);                     % ����ֱ������
sL=length(s);                    % ����źŵĳ���

if fs~=fs1                       % ���������źŵĲ���Ƶ���������Ĳ���Ƶ�ʲ����
    x=resample(data,fs,fs1);     % �������ز�����ʹ��������Ƶ���봿�����źŵĲ���Ƶ����ͬ
else
    x=data;
end

x=x(:);                          % ����������ת����������
x=x-mean(x);                     % ����ֱ������
xL=length(x);                    % ���������ݳ���
if xL>=sL                        % ����������ݳ������ź����ݳ��Ȳ��ȣ����������ݽضϻ���
    x=x(1:sL);
else
    disp('Warning: �������ݶ����ź����ݣ��Բ�0�����㣡')
    x=[x; zeros(sL-xL,1)];
end

Sr=snr;
Es=sum(x.*x);                    % ����źŵ�����
Ev=sum(s.*s);                    % �������������
a=sqrt(Ev/Es/(10^(Sr/10)));      % ����������ı�������
noise=a*x;                       % ���������ķ�ֵ
signal=s+noise;                  % ���ɴ�������

