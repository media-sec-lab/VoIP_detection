function spilt_for_case6(inpath,savepath1,savepath2,savepath3,l)
temp=dir(inpath);
file_length=length(temp);
l_a=0;
l_a2=0;
l_e=0;
for i=3:file_length
    temp_name=temp(i).name;
    index=find(temp_name=='_');
    switch temp_name(index(end-4)+1:index(end-3)-1)
        case {'43','44'}%case6-1
            readname=[inpath,'/',temp_name];
            [x,sampling_rate]=audioread(readname);
            l_a=spilt_speech_to_any_duration(x,savepath1,sampling_rate,l_a,l);
        case {'41','42'}%case6-2
            readname=[inpath,'/',temp_name];
            [x,sampling_rate]=audioread(readname);
            l_a2=spilt_speech_to_any_duration(x,savepath2,sampling_rate,l_a2,l);
        %case{'21','22','31','32','35','36','39','40'}
        case{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18',...
                '19','20','25','26','27','28','29','30','37','38','33','34','23','24','45','46','47','48'}    
            readname=[inpath,'/',temp_name];
            [x2,sampling_rate2]=audioread(readname);
            l_e=spilt_speech_to_any_duration(x2,savepath3,sampling_rate2,l_e,l);
    end
end