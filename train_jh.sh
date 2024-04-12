
# exp 1
VERSION=0
CKPT=bevdepth_full
SAVEPATH=aedet_lss_r50_256x704_128x128_24e_bevdepth


python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 1 \
--ckpt_path ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/ema_12.pth -e \
--key 2 --default_root_dir ./outputs/${SAVEPATH} --bevdepth

mv ./outputs/${SAVEPATH}/results_nusc.json \
./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/results_nusc_${CKPT}_e12.json
mv ./outputs/${SAVEPATH}/metrics_summary.json \
./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e12.json
# python send_result.py ./outputs/${SAVEPATH}/lightning_logs/version_${VERSION}/metrics_summary_${CKPT}_e23.json
