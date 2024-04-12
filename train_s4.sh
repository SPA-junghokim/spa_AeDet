

# # exp 1
VERSION=0
CKPT=memvit
SAVEPATH=aedet_lss_r50_256x704_128x128_24e_cnn

CUDA_VISIBLE_DEVICES=2 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_memvit_multi_scale.py -b 8 --gpus 1 --key 4 --depth_thresh

# mv ./outputs/${SAVEPATH}/results_nusc.json \
# ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e23.json
# mv ./outputs/${SAVEPATH}/metrics_summary.json \
# ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json


# # # exp 1
# VERSION=0
# CKPT=bevdepth_full
# SAVEPATH=aedet_lss_r50_256x704_128x128_24e_bevdepth


# for CKPTN in 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 
# do

#     python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 1 \
#     --ckpt_path ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/ema_${CKPTN}.pth -e \
#     --key 2 --default_root_dir ./outputs/${SAVEPATH} --bevdepth
#     n=${#CKPTN}
#     if [ $n -eq 1 ]
#     then 
#         CKPTN="0${CKPTN}"
#     fi
	
#     #mv outputs/bev_depth_lss_r50_256x704_128x128_24e_2key_ema_7split_da/metrics_summary.json \
#     #outputs/bev_depth_lss_r50_256x704_128x128_24e_2key_ema_7split_da/metrics_summary$CKPT.json

#     mv ./outputs/${SAVEPATH}/results_nusc.json \
#     ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e${CKPTN}.json
#     mv ./outputs/${SAVEPATH}/metrics_summary.json \
#     ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e${CKPTN}.json
#     # python send_result.py ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e8.json

# done

