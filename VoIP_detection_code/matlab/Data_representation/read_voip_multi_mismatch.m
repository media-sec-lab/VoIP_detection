function [mismatch_log_mel_x1,mismatch_log_mel_x2,mismatch_ilog_mel_x1,mismatch_ilog_mel_x2,mismatch_all_wav1,mismatch_all_wav2,mismatch_wav1,mismatch_wav2]=read_voip_multi_mismatch(mismatch_r,mismatch_path1,mismatch_path2,mismatch_num,p,frame_length)

parfor i=1:mismatch_num
    disp(['mismatch index :',num2str(i)])
    mismatch_name1=[mismatch_path1,'/',num2str(mismatch_r(i)),'.wav'];
    warning off
    [samples,fs]=audioread(mismatch_name1);
    wav1=highpass_filter2(samples,1,1)
    [~,log_mel,ilog_mel,~]=get_log_melspec(wav1,floor(fs*frame_length),floor(fs*frame_length),0.5,p,fs);
    log_mel=twoD_norm(log_mel);ilog_mel=twoD_norm(ilog_mel);
    mismatch_log_mel_x1(:,:,i)=log_mel;
    mismatch_ilog_mel_x1(:,:,i)=ilog_mel;
    mismatch_all_wav1(i,:)=wav1;
    mismatch_wav1(:,:,i)=enframe(wav1,hamming(fs*frame_length),floor(fs*frame_length*0.5));

    mismatch_name2=[mismatch_path2,'/',num2str(mismatch_r(i)),'.wav'];
    warning off
    [samples2,fs2]=audioread(mismatch_name2);
    wav2=highpass_filter2(samples2,1,1);
    [~,log_mel2,ilog_mel2,~]=get_log_melspec(wav2,floor(fs2*frame_length),floor(fs2*frame_length),0.5,p,fs2);
    log_mel2=twoD_norm(log_mel2);ilog_mel2=twoD_norm(ilog_mel2);
    mismatch_log_mel_x2(:,:,i)=log_mel2;
    mismatch_ilog_mel_x2(:,:,i)=ilog_mel2;
    mismatch_all_wav2(i,:)=wav2;
    mismatch_wav2(:,:,i)=enframe(wav2,hamming(fs2*frame_length),floor(fs2*frame_length*0.5));
end