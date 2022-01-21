function spilt_for_case2(inpath,savepath1,savepath2,l)
temp=dir(inpath);
file_length=length(temp);
l_a=0;
l_e=0;
for i=3:file_length
    temp_name=temp(i).name;
    index=find(temp_name=='_');
    switch temp_name(index(end-2)+1)
        case 'e'
            readname=[inpath,'/',temp_name];
            [x,sampling_rate]=audioread(readname);
            l_a=spilt_speech_to_any_duration(x,savepath1,sampling_rate,l_a,l);
        otherwise
            readname=[inpath,'/',temp_name];
            [x,sampling_rate]=audioread(readname);
            l_e=spilt_speech_to_any_duration(x,savepath2,sampling_rate,l_e,l);
    end
end