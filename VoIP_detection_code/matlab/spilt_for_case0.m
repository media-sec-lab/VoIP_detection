function spilt_for_case0(inpath,savepath1,l)
temp=dir(inpath);
file_length=length(temp);
n_number=0;
for i=3:file_length
    temp_name=temp(i).name;
    readname=[inpath,'/',temp_name];
    [x,sampling_rate]=audioread(readname);
    n_number=spilt_speech_to_any_duration(x,savepath1,sampling_rate,n_number,l);
end