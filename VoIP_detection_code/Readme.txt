%-----------------------------------------------------------------------------------
%If you have any question about these codes, please feel free to contact huangyuankun2016@email.szu.edu.cn
%
%TODO: in this code we assume that the data follows the directory structure:
% |-VoIP_detection
%   |- VoIP_detection_data
%      |- VPCID_wav
%         |-phone
%         |-voip
%         |-vad
%           |-phone
%           |-voip
%           |-spilt_clips
%             |-case0
%               |-train
%                 |-phone
%                 |-voip
%               |-mat
%             |-case1
%               |-train
%                 |-phone
%                 |-voip
%               |-mismatch
%                 |-phone
%                 |-voip
%               |-mat
%             |-case2
%               |-train
%                 |-phone
%                 |-voip
%               |-mismatch
%                 |-phone
%                 |-voip
%               |-mat
%             |-case3
%               |-train
%                 |-phone
%                 |-voip
%               |-mismatch
%                 |-phone
%                 |-voip
%               |-mat
%             |-case4
%               |-train
%                 |-phone
%                 |-voip
%               |-mismatch
%                 |-phone
%                 |-voip
%               |-mat
%             |-case5
%               |-train
%                 |-phone
%                 |-voip
%               |-mismatch
%                 |-phone
%                 |-voip
%               |-mat
%             |-case6
%               |-train
%                 |-phone
%                 |-voip
%               |-mismatch
%                 |-case6_1
%                   |-phone
%                   |-voip
%                 |-case6_2
%                   |-phone
%                   |-voip
%               |-mat
%   |- VoIP_detection_code
%      |-matlab
%        |-Data_representation
%        |-VAD
%        |-voicebox
%      |-python
%        |-classification
%          |-case6
%          	 |-DNN
%            |-predict
%          |-DNN
%          |-predict
%        |-Evaluation
%          |-case6
%        |-Feature_extraction
%          |-ims_net
%          |-ms_net
%          |-swf_net
%
%------------------------------------------------------------------------------------------------------------------------------
How to use these codes:
First, we download VPCID(VPCID can be download in:https://pan.baidu.com/s/1ZuNbM6Yh8RJ4PFdaWXCgFg , the extracting code is 1111) 
and decode all recordings to waveform(decode the mobile phone recordings to "./VoIP_detection/VoIP_detection_data/VPCID_wav/phone" 
and decode the voip recordings to "./VoIP_detection/VoIP_detection_data/VPCID_wav/voip").
Second, we use matlab codes for data pre-processing and data representation.
Third, we use python codes to train each subnet.
Fourth, we use python codes to extract feature from each trained subnet.
Fifth, we use python to train final NN-based classifier by using the extracted deep features.
Final, we use python to evaluate each trained subnet and the final NN-based classifier.


The steps to use matlab codes for data pre-processing and data representationare described as follows:
Step 1-1: 
   set the parameter (scenario and path) in the main.m in '/VoIP_detection/VoIP_detection_code/matlab/'. 
   The parameter scenario can be set to 'case0','case2','case3','case4','case5','case6'
   For the parameter path, if you download the whole file in path '/x1/x2/x3', then set path to '/x1/x2/x3/VoIP_detection/'
step 1-2: 
   run main.m then the stacked waveform frames, mel-scale spectrogram, and inverted mel-scale spectrogram  
   will be save as train_norm_per_sample_framewav.mat, train_norm_per_sample_logmel.mat, and train_norm_per_sample_ilogmel.mat
   in the corresponding /mat file.
   For example: if scenario=('case1'), then the .mat file will be saved in '/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mat/'
   
   
The steps to use python codes to train each subnet are as follows:
Step 2-1: 
   set the parameter in the python code New_Inception1_fully_conv2.py in '/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ims_net/':
		mat_load_path: the path that used to save the .mat file created in step 1-2. Here we load the .mat file from mat_load_path.
					   For example, if you want to reproduce the experiment of case1, then set mat_load_path='/VoIP_detection_root/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mat'   
		model_save_path: the path used to save the trained IMS-net.  
                         For example, we set model_save_path='/VoIP_detection_root/VoIP_detection_code/python/Feature_extraction/ims_net'
					     then the trained IMS-net model will be saved in '/VoIP_detection_root/VoIP_detection_code/python/Feature_extraction/ims_net/save_model/0.0002/save1_met.ckpt'
Step 2-2:
    run the python code New_Inception1_fully_conv2.py in '/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ims_net/'

					
Step 2-3: 
   set the parameter in the python code New_Inception1_fully_conv2.py in '/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ms_net/':
		mat_load_path: the path that used to save the .mat file created in step 1-2. Here we load the .mat file from mat_load_path.
					   For example, if you want to reproduce the experiment of case1, then set mat_load_path='/VoIP_detection_root/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mat'   
		model_save_path: the path used to save the trained MS-net.  
                         For example, we set model_save_path='/VoIP_detection_root/VoIP_detection_code/python/Feature_extraction/ms_net'
					     then the trained MS-net model will be saved in '/VoIP_detection_root/VoIP_detection_code/python/Feature_extraction/ms_net/save_model/0.0002/save1_met.ckpt'
Step 2-4:
    run the python code New_Inception1_fully_conv2.py in '/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ms_net/'
	
Step 2-5: 
	set the parameter in the python code Inception1_fully_conv_for_frame_wav_with_two_pred.py in '/VoIP_detection/VoIP_detection_code/python/Feature_extraction/swf_net/':
		mat_load_path: the path that used to save the .mat file created in step 1-2. Here we load the .mat file from mat_load_path.
					   For example, if you want to reproduce the experiment of case1, then set mat_load_path='/VoIP_detection_root/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mat'   
		model_save_path: the path used to save the trained SWF-net.  
                         For example, we set model_save_path='/VoIP_detection_root/VoIP_detection_code/python/Feature_extraction/swf_net'
						 then the trained SWF-net model will be saved in '/VoIP_detection_root/VoIP_detection_code/python/Feature_extraction/swf_net/save_model/0.0002/save1_met.ckpt'
Step 2-6:
    run the python code Inception1_fully_conv_for_frame_wav_with_two_pred.py in '/VoIP_detection/VoIP_detection_code/python/Feature_extraction/swf_net/'
	
	
The steps to use python codes to extract feature from the trained each subnet are as follows:
Step 3-1: 
	set the parameter in the python code final_all_predict.py in '/VoIP_detection/VoIP_detection_code/python/classification/':
		mat_load_path: the path that used to save the .mat file created in step 1-2. Here we load the .mat file from mat_load_path.
					   For example, if you want to reproduce the experiment of case1, then set mat_load_path='/VoIP_detection_root/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mat' 
		swf_net_model: the path that used to save the trained SWF-net. Here we need to load the trained SWF-net from this path.
		               we set swf_net_model='/VoIP_detection/VoIP_detection_code/python/Feature_extraction/swf_net'
		ms_net_model: 	the path that used to save the trained MS-net. Here we need to load the trained MS-net from this path.
		               we set ms_net_model='/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ms_net'
		ims_net_model: 	the path that used to save the trained IMS-net. Here we need to load the trained IMS-net from this path.
		               we set ims_net_model='/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ims_net'
		predict_path: the path that used to saved the extracted deep features. Here we saved the predict deep features as train.mat , val.mat , and test.mat in this predict path.
						we set the predict_path to '/VoIP_detection/VoIP_detection_code/python/classification/predict'
Step 3-2: 
	run the python code final_all_predict.py in '/VoIP_detection/VoIP_detection_code/python/classification'

The steps to use python codes to train final NN-based classifier are as follows:
Step 4-1: 
	set the parameter in the python code final_classification_module_1D_CNN.py in '/VoIP_detection/VoIP_detection_code/python/classification/DNN/':
		extracted_feature_path: the path that used to saved the extracted features. 
								Here we load the deep feature by setting extracted_feature_path='/VoIP_detection_root/VoIP_detection_code/python/classification/predict'
		model_save_path: the path that used to saved the trained NN-based classifier. Here we set model_save_path='/VoIP_detection_root/VoIP_detection_code/python/classification/DNN',
						 then the model will be saved in '/VoIP_detection_root/VoIP_detection_code/python/classification/DNN/save_model/0.0002/save1_met.ckpt'
Step 4-2: 			
	run the python code final_classification_module_1D_CNN.py in '/VoIP_detection/VoIP_detection_code/python/classification/DNN/'
	

The steps to use python codes to evaluate each trained subnet and the final NN-based classifier are as follows:
Step 5-1: 
	set the parameter in the python code Evaluation_NN_based_classifier.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/':
		extracted_feature_path: the path that used to saved the extracted features. 
								Here we load the deep feature by setting extracted_feature_path='/VoIP_detection_root/VoIP_detection_code/python/classification/predict'

		model_save_path: the path that used to saved the trained NN-based classifier. 
						 Here we load the trained model by setting model_save_path=('/VoIP_detection/VoIP_detection_code/python/classification/DNN')
						 
Step 5-2:
	run the python code Evaluation_NN_based_classifier.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/' to evaluate the NN-based classifier

Step 5-3: 	
	set the parameter in the python code predict_swf.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/':
		mat_load_path: the path that used to save the .mat file created in step 1-2. Here we load the .mat file from mat_load_path.
					   For example, if you want to evalute the trained SWF-net of case1, then set mat_load_path='/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mat' 
					   
		model_save_path: the path that used to saved the trained SWF-net. 
						 Here we load the trained model by setting model_save_path=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/swf_net')
Step 5-4: 
	run the python code predict_swf.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/' to evaluate the SWF-net.
						
Step 5-5: 	
	set the parameter in the python code predict_ms.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/':
		mat_load_path: the path that used to save the .mat file created in step 1-2. Here we load the .mat file from mat_load_path.
					   For example, if you want to evalute the trained MS-net of case1, then set mat_load_path='/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mat' 
					   
		model_save_path: the path that used to saved the trained MS-net. 
						 Here we load the trained model by setting model_save_path=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ms_net')
Step 5-6: 
	run the python code predict_ms.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/' to evaluate the MS-net.

Step 5-7: 	
	set the parameter in the python code predict_ims.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/':
		mat_load_path: the path that used to save the .mat file created in step 1-2. Here we load the .mat file from mat_load_path.
					   For example, if you want to evalute the trained IMS-net of case1, then set mat_load_path='/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case1/mat' 
					   
		model_save_path: the path that used to saved the trained IMS-net. 
						 Here we load the trained model by setting model_save_path=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ims_net')
Step 5-8: 
	run the python code predict_ims.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/' to evaluate the IMS-net.


PS:
For case 6, the step 3 to step 6 are different:
Step 3-1: 
	set the parameter in the python code final_all_predict.py in '/VoIP_detection/VoIP_detection_code/python/classification/case6':
		mat_load_path: we set mat_load_path='/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mat' 
		swf_net_model: we set swf_net_model='/VoIP_detection/VoIP_detection_code/python/Feature_extraction/swf_net'
		ms_net_model:  we set ms_net_model='/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ms_net'
		ims_net_model: we set ims_net_model='/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ims_net'
		predict_path:  we set the predict_path to '/VoIP_detection/VoIP_detection_code/python/classification/case6/predict'
Step 3-2: 
	run the python code final_all_predict.py in '/VoIP_detection/VoIP_detection_code/python/classification/case6'

Step 4-1: 
	set the parameter in the python code final_classification_module_1D_CNN.py in '/VoIP_detection/VoIP_detection_code/python/classification/case6/DNN/':
		extracted_feature_path: we set extracted_feature_path='/VoIP_detection_root/VoIP_detection_code/python/classification/case6/predict'
		model_save_path: we set model_save_path='/VoIP_detection_root/VoIP_detection_code/python/classification/case6/DNN',
Step 4-2: 			
	run the python code final_classification_module_1D_CNN.py in '/VoIP_detection/VoIP_detection_code/python/classification/case6/DNN/'

Step 5-1: 
	set the parameter in the python code Evaluation_NN_based_classifier.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/case6':
		extracted_feature_path: we load the deep feature by setting extracted_feature_path='/VoIP_detection_root/VoIP_detection_code/python/classification/case6/predict'

		model_save_path: Here we load the trained model by setting model_save_path=('/VoIP_detection/VoIP_detection_code/python/classification/case6/DNN')
						 
Step 5-2:
	run the python code Evaluation_NN_based_classifier.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/case6' to evaluate the NN-based classifier

Step 5-3: 	
	set the parameter in the python code predict_swf.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/case6':
		mat_load_path: we set mat_load_path='/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mat' 
					   
		model_save_path: we load the trained model by setting model_save_path=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/swf_net')
Step 5-4: 
	run the python code predict_swf.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/case6' to evaluate the SWF-net.
						
Step 5-5: 	
	set the parameter in the python code predict_ms.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/case6':
		mat_load_path: we set mat_load_path='/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mat' 
					   
		model_save_path: we load the trained model by setting model_save_path=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ms_net')
Step 5-6: 
	run the python code predict_ms.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/case6' to evaluate the MS-net.

Step 5-7: 	
	set the parameter in the python code predict_ims.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/case6':
		mat_load_path: we set mat_load_path='/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case6/mat' 
					   
		model_save_path: we load the trained model by setting model_save_path=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ims_net')
Step 5-8: 
	run the python code predict_ims.py in '/VoIP_detection/VoIP_detection_code/python/Evaluation/case6' to evaluate the IMS-net.


	














































						
						 