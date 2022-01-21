function main_get_clips(scenario,path,phone_voice_savepath,voip_voice_savepath)
switch(scenario)
    case 'case0'
        %case0(all)
        phone_spilt_savepath=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case0/train/phone');
        voip_spilt_savepath=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case0/train/voip');
        if ~exist(phone_spilt_savepath,'dir')==1
            mkdir(phone_spilt_savepath);
        end
        if ~exist(voip_spilt_savepath,'dir')==1
            mkdir(voip_spilt_savepath);
        end
        l=2;
        spilt_for_case0(phone_voice_savepath,phone_spilt_savepath,l)
        spilt_for_case0(voip_voice_savepath,voip_spilt_savepath,l)
    case 'case1'
        %case1(caller)
        phone_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/train/phone');
        voip_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/train/voip');
        phone_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mismatch/phone');
        voip_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mismatch/voip');
        if ~exist(phone_spilt_savepath_train,'dir')==1
            mkdir(phone_spilt_savepath_train);
        end
        if ~exist(voip_spilt_savepath_train,'dir')==1
            mkdir(voip_spilt_savepath_train);
        end
        if ~exist(phone_spilt_savepath_mismatch,'dir')==1
            mkdir(phone_spilt_savepath_mismatch);
        end
        if ~exist(voip_spilt_savepath_mismatch,'dir')==1
            mkdir(voip_spilt_savepath_mismatch);
        end
        l=2;
        spilt_for_case1(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_train,l)
        spilt_for_case1(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_train,l)
     case 'case2'
        %case2(callee)
        phone_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case2/train/phone');
        voip_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case2/train/voip');
        phone_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case2/mismatch/phone');
        voip_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case2/mismatch/voip');
        if ~exist(phone_spilt_savepath_train,'dir')==1
            mkdir(phone_spilt_savepath_train);
        end
        if ~exist(voip_spilt_savepath_train,'dir')==1
            mkdir(voip_spilt_savepath_train);
        end
        if ~exist(phone_spilt_savepath_mismatch,'dir')==1
            mkdir(phone_spilt_savepath_mismatch);
        end
        if ~exist(voip_spilt_savepath_mismatch,'dir')==1
            mkdir(voip_spilt_savepath_mismatch);
        end
        l=2;
        spilt_for_case2(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_train,l)
        spilt_for_case2(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_train,l)
    case 'case3'
        %case3(speaker)
        phone_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case3/train/phone');
        voip_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case3/train/voip');
        phone_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case3/mismatch/phone');
        voip_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case3/mismatch/voip');
        if ~exist(phone_spilt_savepath_train,'dir')==1
            mkdir(phone_spilt_savepath_train);
        end
        if ~exist(voip_spilt_savepath_train,'dir')==1
            mkdir(voip_spilt_savepath_train);
        end
        if ~exist(phone_spilt_savepath_mismatch,'dir')==1
            mkdir(phone_spilt_savepath_mismatch);
        end
        if ~exist(voip_spilt_savepath_mismatch,'dir')==1
            mkdir(voip_spilt_savepath_mismatch);
        end
        l=2;
        spilt_for_case3(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_train,l)
        spilt_for_case3(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_train,l)
    case 'case4'
        %case4(devices)
        phone_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case4/train/phone');
        voip_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case4/train/voip');
        phone_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case4/mismatch/phone');
        voip_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case4/mismatch/voip');
        if ~exist(phone_spilt_savepath_train,'dir')==1
            mkdir(phone_spilt_savepath_train);
        end
        if ~exist(voip_spilt_savepath_train,'dir')==1
            mkdir(voip_spilt_savepath_train);
        end
        if ~exist(phone_spilt_savepath_mismatch,'dir')==1
            mkdir(phone_spilt_savepath_mismatch);
        end
        if ~exist(voip_spilt_savepath_mismatch,'dir')==1
            mkdir(voip_spilt_savepath_mismatch);
        end
        l=2;
        spilt_for_case4(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_train,l)
        spilt_for_case4(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_train,l)
    case 'case5'
        %case5(software)
        phone_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case5/train/phone');
        voip_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case5/train/voip');
        phone_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case5/mismatch/phone');
        voip_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case5/mismatch/voip');
        if ~exist(phone_spilt_savepath_train,'dir')==1
            mkdir(phone_spilt_savepath_train);
        end
        if ~exist(voip_spilt_savepath_train,'dir')==1
            mkdir(voip_spilt_savepath_train);
        end
        if ~exist(phone_spilt_savepath_mismatch,'dir')==1
            mkdir(phone_spilt_savepath_mismatch);
        end
        if ~exist(voip_spilt_savepath_mismatch,'dir')==1
            mkdir(voip_spilt_savepath_mismatch);
        end
        l=2;
        spilt_for_case5(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_train,l)
        spilt_for_case5(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_train,l)
    case 'case6'
        %case6(caller_brand)
        phone_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/train/phone');
        voip_spilt_savepath_train=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/train/voip');
        phone_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mismatch/case6_1/phone');
        voip_spilt_savepath_mismatch=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mismatch/case6_2/voip');
        phone_spilt_savepath_mismatch2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mismatch/case6_2/phone');
        voip_spilt_savepath_mismatch2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mismatch/case6_2/voip');
        if ~exist(phone_spilt_savepath_train,'dir')==1
            mkdir(phone_spilt_savepath_train);
        end
        if ~exist(voip_spilt_savepath_train,'dir')==1
            mkdir(voip_spilt_savepath_train);
        end
        if ~exist(phone_spilt_savepath_mismatch,'dir')==1
            mkdir(phone_spilt_savepath_mismatch);
        end
        if ~exist(voip_spilt_savepath_mismatch,'dir')==1
            mkdir(voip_spilt_savepath_mismatch);
        end
        if ~exist(phone_spilt_savepath_mismatch2,'dir')==1
            mkdir(phone_spilt_savepath_mismatch2);
        end
        if ~exist(voip_spilt_savepath_mismatch2,'dir')==1
            mkdir(voip_spilt_savepath_mismatch2);
        end
        l=2;
        spilt_for_case6(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_mismatch2,phone_spilt_savepath_train,l)
        spilt_for_case6(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_mismatch2,voip_spilt_savepath_train,l)
end
