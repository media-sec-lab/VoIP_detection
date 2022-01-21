function spilt_for_case4(inpath,savepath1,savepath2,l)
temp=dir(inpath);
file_length=length(temp);
l_a=0;
l_e=0;
for i=3:file_length
    temp_name=temp(i).name;
    index=find(temp_name=='_');
    switch temp_name(index(end-4)+1:index(end-3)-1)
        case {'25','26','27','28'}
            readname=[inpath,'/',temp_name];
            [x,sampling_rate]=audioread(readname);
            l_a=spilt_speech_to_any_duration(x,savepath1,sampling_rate,l_a,l);
        otherwise
            readname=[inpath,'/',temp_name];
            [x2,sampling_rate2]=audioread(readname);
            l_e=spilt_speech_to_any_duration(x2,savepath2,sampling_rate2,l_e,l);
    end
end