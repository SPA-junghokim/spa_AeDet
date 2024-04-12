# BEV Exp 1
VERSION=0
CKPT=nc_pa_ba_bidir


CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis_7split.py --amp_backend native -b 8 --gpus 2 \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_23 \
--neck_consistency \
--neck_consistency_type bilinear \
--use_pa \
--use_ba \
--consis_bidirec

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis_7split.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split/lightning_logs/version_${VERSION}/ema_23.pth -e \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_23 \
--neck_consistency \
--neck_consistency_type bilinear \
--use_pa \
--use_ba \
--consis_bidirec

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_23/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_23/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_23/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_23/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_23/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json


