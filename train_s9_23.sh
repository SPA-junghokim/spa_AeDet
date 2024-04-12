# BEV Exp 1
VERSION=0
CKPT=key2

CUDA_VISIBLE_DEVICES=2,3 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar.py --amp_backend native -b 4 --gpus 2 --key 2

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar.py -b 6 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}/ema_23.pth -e

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json




# BEV Exp 1
VERSION=2
CKPT=key2_deform

CUDA_VISIBLE_DEVICES=2,3 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar.py --amp_backend native -b 4 --gpus 2 --key 2 --deform_type deform

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar.py -b 6 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}/ema_23.pth -e \
--deform_type deform

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json




# BEV Exp 1
VERSION=4
CKPT=polar_nc

CUDA_VISIBLE_DEVICES=2,3 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar.py --amp_backend native -b 4 --gpus 2 --neck_consistency

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar.py -b 6 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}/ema_23.pth -e \
--neck_consistency

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json


# BEV Exp 1
VERSION=6
CKPT=polar_nc_deform

CUDA_VISIBLE_DEVICES=2,3 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar.py --amp_backend native -b 4 --gpus 2 --neck_consistency --deform_type deform

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar.py -b 6 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}/ema_23.pth -e \
--neck_consistency \
--deform_type deform

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json

