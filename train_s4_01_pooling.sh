# # BEV Exp 1
# VERSION=0
# CKPT=pooling

# CUDA_VISIBLE_DEVICES=0,1 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py --amp_backend native -b 4 --gpus 2 \
# --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
# --pooling_consistency 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 2 \
# --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ema_23.pth -e \
# --pooling_consistency 

# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json




# # BEV Exp 1
# VERSION=2
# CKPT=pooling_pa

# CUDA_VISIBLE_DEVICES=0,1 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py --amp_backend native -b 4 --gpus 2 \
# --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
# --pool_pa \
# --pooling_consistency 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 2 \
# --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ema_23.pth -e \
# --pool_pa \
# --pooling_consistency 

# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json






# # BEV Exp 1
# VERSION=4
# CKPT=pooling_ba

# CUDA_VISIBLE_DEVICES=0,1 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py --amp_backend native -b 4 --gpus 2 \
# --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
# --pool_ba \
# --pooling_consistency 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 2 \
# --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ema_23.pth -e \
# --pool_ba \
# --pooling_consistency 

# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json








# # BEV Exp 1
# VERSION=6
# CKPT=pooling_pa_ba

# CUDA_VISIBLE_DEVICES=0,1 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py --amp_backend native -b 4 --gpus 2 \
# --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
# --pool_pa \
# --pool_ba \
# --pooling_consistency 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 2 \
# --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ema_23.pth -e \
# --pool_pa \
# --pool_ba \
# --pooling_consistency 

# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}/ \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/results_nusc_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e/metrics_summary_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json








# BEV Exp 1
VERSION=30
CKPT=pooling_pamlp

CUDA_VISIBLE_DEVICES=0,1 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py --amp_backend native -b 4 --gpus 2 \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
--pool_pa_mlp \
--pooling_consistency 

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 2 \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}/ema_23.pth -e \
--pool_pa_mlp \
--pooling_consistency 

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json







# BEV Exp 1
VERSION=32
CKPT=pooling_pamlp_poolbd

CUDA_VISIBLE_DEVICES=0,1 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py --amp_backend native -b 4 --gpus 2 \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
--pool_pa_mlp \
--pool_bev_detach \
--pooling_consistency 

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 2 \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}/ema_23.pth -e \
--pool_pa_mlp \
--pool_bev_detach \
--pooling_consistency 

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json




# BEV Exp 1
VERSION=34
CKPT=pooling_bamlp_poolpd

CUDA_VISIBLE_DEVICES=0,1 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py --amp_backend native -b 4 --gpus 2 \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
--pool_ba_mlp \
--pool_pers_detach \
--pooling_consistency 

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 2 \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}/ema_23.pth -e \
--pool_ba_mlp \
--pool_pers_detach \
--pooling_consistency 

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json





# BEV Exp 1
VERSION=36
CKPT=pooling_pamlp_bamlp

CUDA_VISIBLE_DEVICES=0,1 python exps/aedet/aedet_lss_r50_256x704_128x128_24e_consis.py --amp_backend native -b 4 --gpus 2 \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
--pool_pa_mlp \
--pool_ba_mlp \
--pooling_consistency 

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 2 \
--default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}/ema_23.pth -e \
--pool_pa_mlp \
--pool_ba_mlp \
--pooling_consistency 

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_consis_pool_consis/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json
