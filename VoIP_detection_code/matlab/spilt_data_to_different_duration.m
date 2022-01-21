function spilt_data_to_different_duration(case_type,phone_voice_savepath,voip_voice_savepath,phone_spilt_savepath_train,voip_spilt_savepath_train,phone_spilt_savepath_mismatch,voip_spilt_savepath_mismatch,phone_spilt_savepath_mismatch2,voip_spilt_savepath_mismatch2,l)
if nargin == 6, 
    phone_spilt_savepath_mismatch=('');
    voip_spilt_savepath_mismatch=('');
    phone_spilt_savepath_mismatch2=('');
    voip_spilt_savepath_mismatch2=('');
elseif nargin==8
    phone_spilt_savepath_mismatch2=('');
    voip_spilt_savepath_mismatch2=('');
end
    
switch case_type
    case{'case0'}
        spilt_for_case0(phone_voice_savepath,phone_spilt_savepath_train,l)
        spilt_for_case0(voip_voice_savepath,voip_spilt_savepath_train,l)
    case{'case1'}
        spilt_for_case1(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_train,l)
        spilt_for_case1(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_train,l)
    case{'case2'}
        spilt_for_case2(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_train,l)
        spilt_for_case2(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_train,l)
    case{'case3'}
        spilt_for_case3(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_train,l)
        spilt_for_case3(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_train,l)
    case{'case4'}
        spilt_for_case4(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_train,l)
        spilt_for_case4(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_train,l)
    case{'case5'}
        spilt_for_case5(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_train,l)
        spilt_for_case5(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_train,l)
    case{'case6'}
        spilt_for_case6(phone_voice_savepath,phone_spilt_savepath_mismatch,phone_spilt_savepath_mismatch2,phone_spilt_savepath_train,l)
        spilt_for_case6(voip_voice_savepath,voip_spilt_savepath_mismatch,voip_spilt_savepath_mismatch2,voip_spilt_savepath_train,l)
end