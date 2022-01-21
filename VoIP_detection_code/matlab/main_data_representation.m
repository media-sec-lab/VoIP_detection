function main_data_representation(scenario,path)
switch scenario
    case 'case0'
        %case0(all)
        filepath1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case0/train/phone');
        filepath2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case0/train/voip');
        mismatch_path1=filepath1;
        mismatch_path2=filepath2;
        train_rate=0.8*0.8;
        val_rate=0.8*0.2;
        datanum=47500;
        mismatch_num=0;
        save_path=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case0/mat');
        p=48;
        frame_length=0.03;
        read_voip_for_resampling(filepath1,filepath2,mismatch_path1,mismatch_path2,train_rate,val_rate,datanum,mismatch_num,save_path,p,frame_length)
        clear
    case 'case1'
        %case1(caller location)
        filepath1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/train/phone')
        filepath2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/train/voip')
        mismatch_path1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mismatch/phone')
        mismatch_path2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mismatch/voip')
        train_rate=0.8
        val_rate=0.2
        datanum=35000
        mismatch_num=12000
        save_path=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mat')
        p=48
        frame_length=0.03
        read_voip_for_resampling(filepath1,filepath2,mismatch_path1,mismatch_path2,train_rate,val_rate,datanum,mismatch_num,save_path,p,frame_length)
        clear
    case 'case2'
        %case2(callee location)
        filepath1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case2/train/phone')
        filepath2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case2/train/voip')
        mismatch_path1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case2/mismatch/phone')
        mismatch_path2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case2/mismatch/voip')
        train_rate=0.8
        val_rate=0.2
        datanum=23000
        mismatch_num=24000
        save_path=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case2/mat')
        p=48
        frame_length=0.03
        read_voip_for_resampling(filepath1,filepath2,mismatch_path1,mismatch_path2,train_rate,val_rate,datanum,mismatch_num,save_path,p,frame_length)
        clear
    case 'case3'
        %case3(speaker)
        filepath1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case3/train/phone')
        filepath2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case3/train/voip')
        mismatch_path1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case3/mismatch/phone')
        mismatch_path2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case3/mismatch/voip')
        train_rate=0.8
        val_rate=0.2
        datanum=46000
        mismatch_num=1700
        save_path=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case3/mat')
        p=48
        frame_length=0.03
        read_voip_for_resampling(filepath1,filepath2,mismatch_path1,mismatch_path2,train_rate,val_rate,datanum,mismatch_num,save_path,p,frame_length)
        clear
    case 'case4'
        %case4(devices)
        filepath1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case4/train/phone')
        filepath2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case4/train/voip')
        mismatch_path1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case4/mismatch/phone')
        mismatch_path2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case4/mismatch/voip')
        train_rate=0.8
        val_rate=0.2
        datanum=44000
        mismatch_num=3450
        save_path=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case4/mat')
        p=48
        frame_length=0.03
        read_voip_for_resampling(filepath1,filepath2,mismatch_path1,mismatch_path2,train_rate,val_rate,datanum,mismatch_num,save_path,p,frame_length)
        clear
    case 'case5'
        %case5(software)
        filepath1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case5/train/phone')
        filepath2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case5/train/voip')
        mismatch_path1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case5/mismatch/phone')
        mismatch_path2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case5/mismatch/voip')
        train_rate=0.8
        val_rate=0.2
        datanum=40000
        mismatch_num=7000
        save_path=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case5/mat')
        p=48
        frame_length=0.03
        read_voip_for_resampling(filepath1,filepath2,mismatch_path1,mismatch_path2,train_rate,val_rate,datanum,mismatch_num,save_path,p,frame_length)
        clear
    case 'case6'
        %%
        %case6(caller_brand)
        filepath1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/train/phone')
        filepath2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/train/voip')
        mismatch_path1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mismatch/case6_1/phone')
        mismatch_path2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mismatch/case6_1/voip')
        mismatch2_path1=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mismatch/case6_2/phone')
        mismatch2_path2=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mismatch/case6_2/voip')
        train_rate=0.8
        val_rate=0.2
        datanum=47000
        mismatch_num=1400
        mismatch_num2=1400
        save_path=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mat')
        p=48
        frame_length=0.03
        read_voip_for_resampling_20100112_for_2_multi_vad(filepath1,filepath2,mismatch_path1,mismatch_path2,mismatch2_path1,mismatch2_path2,...
            train_rate,val_rate,datanum,mismatch_num,mismatch_num2,save_path,p,frame_length)
        clear
end