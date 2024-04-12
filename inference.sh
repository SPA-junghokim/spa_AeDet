


# BEV Exp 1
#VERSION=0
#CKPT=da_origin_from_consisfile


#python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py -b 4 --gpus 1 \
#--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4/${CKPT}/ema_23.pth -e \
#--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4 \
#--use_da_origin

#mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/results_nusc.json \
#./outputs/aedet_lss_r50_256x704_128x128_24e_consis/results_nusc_${CKPT}.json
#mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/metrics_summary.json \
#./outputs/aedet_lss_r50_256x704_128x128_24e_consis/metrics_summary_${CKPT}.json
#mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}/ \
#./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}_${CKPT}/
#mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/results_nusc_${CKPT}.json \
#./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
#mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis/metrics_summary_${CKPT}.json \
#./outputs/aedet_lss_r50_256x704_128x128_24e_consis/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json





# BEV Exp 1
VERSION=2
CKPT=key2_from_consisfile


python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py -b 4 --gpus 1 --key 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4/${CKPT}/ema_23.pth -e \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis

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






# BEV Exp 1
VERSION=6
CKPT=bevdepth

python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 1  --key 2 --bevdepth \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4/${CKPT}/ema_23.pth -e 

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
VERSION=8
CKPT=aedet_layer_norm

python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 1 --layer_norm --key 2 \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4/${CKPT}/ema_23.pth -e 

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
VERSION=10
CKPT=bc_swap

python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py -b 4 --gpus 1 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4/${CKPT}/ema_23.pth -e \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4 \
--bev_consistency \
--swap_bev_xy \
--swap_pers_uv

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



# BEV Exp 1
VERSION=10
CKPT=nc_swap

python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py -b 4 --gpus 1 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4/${CKPT}/ema_23.pth -e \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_7split_batch4 \
--neck_consistency \
--swap_bev_xy \
--swap_pers_uv

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


