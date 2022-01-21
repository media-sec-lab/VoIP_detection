function vad_myself_read_wav(filename,savepath)

IS=0.25;                                % 设置前导无话段长度
%wlen=200;                               % 设置帧长为25ms
%inc=80;                                 % 求帧移

[xx,fs]=audioread(filename);                   % 读入数据
xx=xx(:,1);
xx1=xx;
wlen=0.025*fs;
inc=fs*0.01;
xx=xx-mean(xx);                         % 消除直流分量
x=xx/max(abs(xx));                      % 幅值归一化
N=length(x);                            % 取信号长度
time=(0:N-1)/fs;                        % 设置时间
signal=x;                % 叠加噪声

wnd=hamming(wlen);                      % 设置窗函数
overlap=wlen-inc;                       % 求重叠区长度
NIS=fix((IS*fs-wlen)/inc +1);           % 求前导无话段帧数

y=enframe(signal,wnd,inc)';             % 分帧
fn=size(y,2);                           % 求帧数
frameTime=frame2time(fn, wlen, inc, fs);% 计算各帧对应的时间

Y=fft(y);                               % FFT变换
N2=wlen/2+1;                            % 取正频率部分
n2=1:N2;
Y_abs=abs(Y(n2,:));                     % 取幅值

for k=1:fn                              % 计算每帧的频带方差
    Dvar(k)=var(Y_abs(:,k))+eps;
end
dth=mean(Dvar(1:NIS));                  % 求取阈值
T1=1.5*dth;
T2=3*dth;
[voiceseg,vsl,SF,NF,flag]=vad_param1D(Dvar,T1,T2);% 频域方差双门限的端点检测
if flag==0
    split_begin=voiceseg(1).begin;
    temp_number=0;
    wav_name_index=find(filename=='/');
    wav_name=filename(wav_name_index(end)+1:length(filename));
    new_name_1=wav_name(1:find(wav_name=='.')-1);
    for k=1 : vsl-1                           % 标出语音端点
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