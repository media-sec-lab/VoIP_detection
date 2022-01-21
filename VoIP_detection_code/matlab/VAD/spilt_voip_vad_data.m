function spilt_voip_vad_data(inpath,savepath)
temp=dir(fullfile(inpath,'*.wav'));
temp={temp.name}';
file_length=length(temp);
for i=1:file_length
    temp_name=temp{i};
    readname=[inpath,'/',temp_name];
    vad_myself_read_wav(readname,savepath)
end