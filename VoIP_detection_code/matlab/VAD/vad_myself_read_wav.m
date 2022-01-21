function vad_myself_read_wav(filename,savepath)

IS=0.25;                                % ����ǰ���޻��γ���
%wlen=200;                               % ����֡��Ϊ25ms
%inc=80;                                 % ��֡��

[xx,fs]=audioread(filename);                   % ��������
xx=xx(:,1);
xx1=xx;
wlen=0.025*fs;
inc=fs*0.01;
xx=xx-mean(xx);                         % ����ֱ������
x=xx/max(abs(xx));                      % ��ֵ��һ��
N=length(x);                            % ȡ�źų���
time=(0:N-1)/fs;                        % ����ʱ��
signal=x;                % ��������

wnd=hamming(wlen);                      % ���ô�����
overlap=wlen-inc;                       % ���ص�������
NIS=fix((IS*fs-wlen)/inc +1);           % ��ǰ���޻���֡��

y=enframe(signal,wnd,inc)';             % ��֡
fn=size(y,2);                           % ��֡��
frameTime=frame2time(fn, wlen, inc, fs);% �����֡��Ӧ��ʱ��

Y=fft(y);                               % FFT�任
N2=wlen/2+1;                            % ȡ��Ƶ�ʲ���
n2=1:N2;
Y_abs=abs(Y(n2,:));                     % ȡ��ֵ

for k=1:fn                              % ����ÿ֡��Ƶ������
    Dvar(k)=var(Y_abs(:,k))+eps;
end
dth=mean(Dvar(1:NIS));                  % ��ȡ��ֵ
T1=1.5*dth;
T2=3*dth;
[voiceseg,vsl,SF,NF,flag]=vad_param1D(Dvar,T1,T2);% Ƶ�򷽲�˫���޵Ķ˵���
if flag==0
    split_begin=voiceseg(1).begin;
    temp_number=0;
    wav_name_index=find(filename=='/');
    wav_name=filename(wav_name_index(end)+1:length(filename));
    new_name_1=wav_name(1:find(wav_name=='.')-1);
    for k=1 : vsl-1                           % ��������˵�
        nx1=voiceseg(k+1).begin; nx2=voiceseg(k).end;
        if (frameTime(nx1)-frameTime(nx2)) >= 0.25
            split_end=voiceseg(k).end;
            temp=xx1(frameTime(split_begin)*fs+1:frameTime(split_end)*fs);
            temp_number=temp_number+1;
            save_name=([savepath,'/',new_name_1,'_',num2str(temp_number),'.wav']);
            audiowrite(save_name,temp,fs)
            split_begin=voiceseg(k+1).begin;
        elseif (frameTime(nx1)-frameTime(nx2)) < 0.25
            if k==(vsl-1)
                split_end=voiceseg(k+1).end;
                temp=xx1(frameTime(split_begin)*fs+1:frameTime(split_end)*fs);
                temp_number=temp_number+1;
                save_name=([savepath,'/',new_name_1,'_',num2str(temp_number),'.wav']);
                audiowrite(save_name,temp,fs)
            end
        end
    end
else
    wav_name_index=find(filename=='/');
    wav_name=filename(wav_name_index(end)+1:length(filename));
    new_name_1=wav_name(1:find(wav_name=='.')-1);
    save_name=([savepath,'/',new_name_1,'.wav']);
    audiowrite(save_name,xx1,fs)
end