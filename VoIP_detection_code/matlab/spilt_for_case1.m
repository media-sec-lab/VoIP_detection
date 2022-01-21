function spilt_for_case1(inpath,savepath1,savepath2,l)
temp=dir(inpath);
file_length=length(temp);
l_a=0;
l_others=0;
for i=3:file_length
    temp_name=temp(i).name;
    index=find(temp_name=='_');
    switch temp_name(index(end-3)+1)
        case 'a'
            readname=[inpath,'/',temp_name];
            [x,sampling_rate]=audioread(readname);
            l_a=spilt_speech_to_any_duration(x,savepath1,sampling_rate,l_a,l);
            %l_a=spilt_to2s_data_20180612(x,savepath1,sampling_rate,l_a);
        otherwise
            readname=[inpath,'/',temp_name];
            [x,sampling_rate]=audioread(readname);
            l_others=spilt_speech_to_any_duration(x,savepath2,sampling_rate,l_others,l);
            %l_e=spilt_to2s_data_20180612(x,savepath2,sampling_rate,l_e);
    end
end