

# # exp 1
VERSION=0
CKPT=b8_g2_key4
SAVEPATH=aedet_lss_r50_256x704_128x128_24e_memvit_ms_dcl_da_bevdepth_4key_baseline

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_memvit_ms_dcl_da.py -b 8 --gpus 2 --key 4 --default_root_dir ./outputs/${SAVEPATH}

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_memvit_ms_dcl_da.py -b 8 --gpus 2 --key 4 \
--default_root_dir ./outputs/${SAVEPATH} --bevdepth \
--ckpt_path ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/ema_23.pth -e


mv ./outputs/${SAVEPATH}/results_nusc.json \
./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e23.json
mv ./outputs/${SAVEPATH}/metrics_summary.json \
./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json
python send_result.py ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json

