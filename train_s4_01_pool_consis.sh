
# Exp 2
# VERSION=0
# CKPT=consis_pool_nega_cos

# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py --amp_backend native -b 4 --gpus 2 \
# --pooling_consistency --pool_bev_detach

# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py -b 4 --gpus 2 \
# --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}/ema_23.pth -e \
# --pooling_consistency --pool_bev_detach

# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/results_nusc.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/metrics_summary.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/metrics_summary_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}/ \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}_${CKPT}/
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/results_nusc_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/metrics_summary_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



# Exp 4
VERSION=6
CKPT=consis_poolv2_infonce

CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py --amp_backend native -b 4 --gpus 2 \
--pooling_consistency_v2 --pool_loss infonce --pool_bev_detach

CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py -b 4 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}/ema_23.pth -e \
--pooling_consistency_v2 --pool_loss infonce --pool_bev_detach

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json

