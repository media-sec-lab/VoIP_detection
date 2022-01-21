function spilt_for_case5(inpath,savepath1,savepath2,l)
temp=dir(inpath);
file_length=length(temp);
l_skype=0;
l_other=0;
for i=3:file_length
    temp_name=temp(i).name;
    index=find(temp_name=='_');
    switch temp_name(index(end-4)+1:index(end-3)-1)
        case {'25','26','27','28','29','30','23','24'}
            readname=[inpath,'/',temp_name];
            [x,sampling_rate]=audioread(readname);
            l_skype=spilt_speech_to_any_duration(x,savepath1,sampling_rate,l_skype,l);
        otherwise
            readname=[inpath,'/',temp_name];
            [x,sampling_rate]=audioread(readname);
            l_other=spilt_speech_to_any_duration(x,savepath2,sampling_rate,l_other,l);
    end
end