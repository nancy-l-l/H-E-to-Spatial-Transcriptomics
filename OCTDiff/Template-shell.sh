#train    !change path!  
python3 main.py --config configs/OCTDiff.yaml --train --sample_at_start --save_top --gpu_ids 1 #run this run the model

#test (inference)    !change path!  
python3 main.py --config configs/OCTDiff.yaml --sample_to_eval --gpu_ids 1 --resume_model /data/yetian/OCTDiff/results/ANA_Augmented/BrownianBridge/checkpoint/latest_model_500.pth