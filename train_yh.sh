






CUDA_VISIBLE_DEVICES=0 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_multi_scale_mm_2stage.py --amp_backend native -b 4 --gpus 1 --bevdepth


# VERSION=0
# CKPT=position_aware_frustum
# python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_spn.py --amp_backend native -b 8 --gpus 2 --bevdepth --use_4split

# python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_spn.py -b 8 --gpus 2 \
# --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/lightning_logs/version_${VERSION}/ema_23.pth -e --bevdepth

# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/results_nusc.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/metrics_summary.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/metrics_summary_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/lightning_logs/version_${VERSION}/ \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/lightning_logs/version_${VERSION}_${CKPT}/
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/results_nusc_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/metrics_summary_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_spn/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json


