%------------------------Parameter------------------------------
%phone_VPCID_wav_path      The path used to save the decoded .wav files of mobile phone call recording of VPCID                   
%phone_voice_savepath      The path used to save the voiced speech of mobile phone call recording of VPCID
%voip_VPCID_wav_path       The path used to save the decoded .wav files of VoIP call recording of VPCID   
%voip_voice_savepath       The path used to save the voiced speech of VoIP call recording of VPCID
%path                      The home path 
%scenario                  The case selected for experiments
%-------------------------------------------------------
%%
%VAD
path=('/VoIP_detection/');
addpath(genpath(path))
phone_VPCID_wav_path=fullfile(path,'/VoIP_detection_data/VPCID_wav/phone');
phone_voice_savepath=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/phone');
if ~exist(phone_voice_savepath,'dir')==1
    mkdir(phone_voice_savepath);
end
spilt_voip_vad_data(phone_VPCID_wav_path,phone_voice_savepath)
voip_VPCID_wav_path=fullfile(path,'/VoIP_detection_data/VPCID_wav/voip');
voip_voice_savepath=fullfile(path,'/VoIP_detection_data/VPCID_wav/vad/voip');
if ~exist(voip_voice_savepath,'dir')==1
    mkdir(voip_voice_savepath);
end
spilt_voip_vad_data(voip_VPCID_wav_path,voip_voice_savepath)
%%
scenario=('case0');
main_get_clips(scenario,path,phone_voice_savepath,voip_voice_savepath)
main_data_representation(scenario,path)