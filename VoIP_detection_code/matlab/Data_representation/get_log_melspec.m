function [mel,log_mel,i_mel,i_log_mel,f_mel,f_log_mel,t,log_t]=get_log_melspec(x,frame_length,fft_length,overlap_rate,p,fs)
addpath(genpath('E:\learning lesson\������\voicebox'));%
bank=melbankm(p,fft_length,fs,0,0.5,'m');
bank=full(bank);
bank=bank/max(bank(:));
i_bank=zeros(size(bank));
for i=1:p
    for j=1:(fft_length/2-1)
        i_bank(i,j)=bank(p-i+1,(fft_length/2-1)-j+1);
    end
end
%x=highpass_filter(x,0);
xx=enframe(x,hamming(frame_length),frame_length*overlap_rate);
%xx=enframe(x,hamming(fft_length));
xx=xx';
xx_f=abs(fft(xx,fft_length));
size(xx_f)
t=xx_f(1:fft_length/2+1,:);
t=t.^2;
log_t=log(t);
log_t(find(isinf(log_t)==1))=0;
mel=bank*t;
i_mel=i_bank*t;
i_log_mel=log(i_mel);
i_log_mel(find(isinf(i_log_mel)==1))=0;
log_mel=log(mel);
log_mel(find(isinf(log_mel)==1))=0;
f_bank=melbankm(p,fft_length,fs,0,0.5,'f');
f_bank=full(f_bank);
f_bank=f_bank/max(f_bank(:));
f_mel=f_bank*t;
f_log_mel=log(f_mel);
f_log_mel(find(isinf(f_log_mel)==1))=0;
