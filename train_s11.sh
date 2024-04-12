


# exp 1
VERSION=0
CKPT=bevdepth_full_4key
SAVEPATH=aedet_lss_r50_256x704_128x128_24e_memvit_ms_dcl_da


# python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_memvit_ms_dcl_da.py --amp_backend native -b 7 --gpus 4 \
# --key 4 --default_root_dir ./outputs/${SAVEPATH} --max_epochs 30 --bevdepth True

python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_memvit_ms_dcl_da.py -b 7 --gpus 4 \
--ckpt_path ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/ema_23.pth -e \
--key 4 --default_root_dir ./outputs/${SAVEPATH} --bevdepth True

mv ./outputs/${SAVEPATH}/results_nusc.json \
./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e23.json
mv ./outputs/${SAVEPATH}/metrics_summary.json \
./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json
python send_result.py ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json


python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 7 --gpus 4 \
--ckpt_path ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/ema_29.pth -e \
--key 4 --default_root_dir ./outputs/${SAVEPATH} --bevdepth True

mv ./outputs/${SAVEPATH}/results_nusc.json \
./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e29.json
mv ./outputs/${SAVEPATH}/metrics_summary.json \
./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e29.json
python send_result.py ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e29.json




# # # exp 1
# VERSION=0
# CKPT=bevdepth_full_4key_mm
# SAVEPATH=aedet_lss_r50_256x704_128x128_24e_mm_bevdepth_4key

# python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py --amp_backend native -b 7 --gpus 4 \
# --key 4 --default_root_dir ./outputs/${SAVEPATH} --max_epochs 30 --bevdepth

# python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py -b 7 --gpus 4 \
# --ckpt_path ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/ema_23.pth -e \
# --key 4 --default_root_dir ./outputs/${SAVEPATH} --bevdepth

# mv ./outputs/${SAVEPATH}/results_nusc.json \
# ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e23.json
# mv ./outputs/${SAVEPATH}/metrics_summary.json \
# ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json
# python send_result.py ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json


# python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py -b 7 --gpus 4 \
# --ckpt_path ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/ema_29.pth -e \
# --key 4 --default_root_dir ./outputs/${SAVEPATH} --bevdepth

# mv ./outputs/${SAVEPATH}/results_nusc.json \
# ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e29.json
# mv ./outputs/${SAVEPATH}/metrics_summary.json \
# ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e29.json
# python send_result.py ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e29.json




# # exp 1
# VERSION=0
# CKPT=bevdepth_full_4key_cnn_temp_agg
# SAVEPATH=aedet_lss_r50_256x704_128x128_24e_memvit

# python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_memvit.py -b 8 --gpus 4 \
# --ckpt_path ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/ema_23.pth -e \
# --key 4 --default_root_dir ./outputs/${SAVEPATH}

# mv ./outputs/${SAVEPATH}/results_nusc.json \
# ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e23.json
# mv ./outputs/${SAVEPATH}/metrics_summary.json \
# ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json
# python send_result.py ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json

# python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_memvit.py --amp_backend native -b 8 --gpus 4 \
# --key 4 --default_root_dir ./outputs/${SAVEPATH} --max_epochs 30 --ckpt_path outputs/aedet_lss_r50_256x704_128x128_24e_memvit/lightning_logs/version_0/origin_23.pth


# CKPT=bevdepth_full_4key_cnn_temp_agg
# SAVEPATH=aedet_lss_r50_256x704_128x128_24e_memvit_multi_scale
# VERSION=2

# python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_memvit_multi_scale.py -b 8 --gpus 4 \
# --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_memvit_multi_scale/lightning_logs/version_2/ema_29.pth -e \
# --key 4 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_memvit

# mv ./outputs/${SAVEPATH}/results_nusc.json \
# ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e29.json
# mv ./outputs/${SAVEPATH}/metrics_summary.json \
# ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e29.json
# python send_result.py ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e29.json

