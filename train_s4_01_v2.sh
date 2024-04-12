
# # exp 1
VERSION=0
CKPT=b8_g2_key4
SAVEPATH=aedet_lss_r50_256x704_128x128_24e_bevdepth_batch_past

# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 6 --gpus 2 --key 4 --use_4split --past_batch --bevdepth \
# --default_root_dir ./outputs/${SAVEPATH}


CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 6 --gpus 2 --key 4 --past_batch --bevdepth \
--default_root_dir ./outputs/${SAVEPATH} \
--ckpt_path ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/ema_23.pth -e


mv ./outputs/${SAVEPATH}/results_nusc.json \
./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e23.json
mv ./outputs/${SAVEPATH}/metrics_summary.json \
./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json
python send_result.py ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json
