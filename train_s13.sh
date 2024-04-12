# BEV Exp 1
VERSION=0
CKPT=1layer

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py --amp_backend native -b 7 --gpus 4 \
--full --convlstm_layer 1 --deform_conv_lstm --motion_gate

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py -b 7 --gpus 4 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_mm/lightning_logs/version_${VERSION}/ema_23.pth -e \
--full --convlstm_layer 1 --deform_conv_lstm --motion_gate

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_12/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_12/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_12/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_12/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json







# BEV Exp 1
VERSION=2
CKPT=no_residual

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py --amp_backend native -b 7 --gpus 4 \
--full --mm_no_residual --deform_conv_lstm --motion_gate

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py -b 7 --gpus 4 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_mm/lightning_logs/version_${VERSION}/ema_23.pth -e \
--full --mm_no_residual --deform_conv_lstm --motion_gate

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_12/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_12/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_12/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_12/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json