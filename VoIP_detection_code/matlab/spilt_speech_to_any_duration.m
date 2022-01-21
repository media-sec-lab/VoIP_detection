function l=spilt_speech_to_any_duration(x,savepath,sampling_rate,exist_number,spilt_length)
l=exist_number;
j=numel(x)/sampling_rate;
num=floor(j/spilt_length);
for k=1:num
    b=x((k-1)*sampling_rate*spilt_length+1:k*sampling_rate*spilt_length);
    %the following row for normalization
    %standardVar = std(b(:));
    %if (standardVar~=0) 
    %    out1=(b-mean(b(:)))/standardVar;
    %else
    %    out1=b;
    %end
    b_save=b;
    
    l=l+1;
    audiowrite([savepath,'/',num2str(l),'.wav'],b_save,sampling_rate);
end