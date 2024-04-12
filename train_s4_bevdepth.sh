# BEV Exp 1
VERSION=0
CKPT=bevdepth

CUDA_VISIBLE_DEVICES=0,1,2,3 python exps/aedet/aedet_lss_r50_256x704_128x128_24e.py --amp_backend native -b 8 --gpus 4 --key 2 --full --bevdepth
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 8 --gpus 4  --key 2 --bevdepth \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ema_23.pth -e \

mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



# BEV Exp 1
VERSION=2
CKPT=layer_norm

CUDA_VISIBLE_DEVICES=0,1,2,3 python exps/aedet/aedet_lss_r50_256x704_128x128_24e.py --amp_backend native -b 8 --gpus 4 --key 2 --full --layer_norm 
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 8 --gpus 4 --layer_norm --key 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ema_23.pth -e \

mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json

