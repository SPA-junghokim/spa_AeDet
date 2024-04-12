

#exp 1
# VERSION=2
# CKPT=1key_baseline_4batch

# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_7split.py --amp_backend native -b 4 --gpus 2 


# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_7split.py -b 4 --gpus 2 \
# --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/lightning_logs/version_${VERSION}/ema_23.pth -e


# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/results_nusc.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/metrics_summary.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/metrics_summary_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/lightning_logs/version_${VERSION}/ \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/lightning_logs/version_${VERSION}_${CKPT}/
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/results_nusc_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/metrics_summary_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_7split/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



#exp 2
# VERSION=2
# CKPT=2key_baseline_4batch

# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_2key_7split.py --amp_backend native -b 4 --gpus 2 


# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_2key_7split.py -b 4 --gpus 2 \
# --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/lightning_logs/version_${VERSION}/ema_23.pth -e


# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/results_nusc.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/metrics_summary.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/metrics_summary_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/lightning_logs/version_${VERSION}/ \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/lightning_logs/version_${VERSION}_${CKPT}/
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/results_nusc_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/metrics_summary_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json


#exp 3
# VERSION=0
# CKPT=2key_baseline_mm_4batch

# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm.py --amp_backend native -b 3 --gpus 2 

# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm.py -b 3 --gpus 2 \
# --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/lightning_logs/version_${VERSION}/ema_23.pth -e

# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/results_nusc.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/metrics_summary.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/metrics_summary_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/lightning_logs/version_${VERSION}/ \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/lightning_logs/version_${VERSION}_${CKPT}/
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/results_nusc_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/metrics_summary_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_7split_mm/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json




# #exp 4
# VERSION=72
# CKPT=polarbev_nmTFF

# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 4 --gpus 2\
# --vel_div mul --norm_bbox

# CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 4 --gpus 2 \
# --vel_div mul --norm_bbox --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}/ema_23.pth -e 

# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/results_nusc.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/metrics_summary.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/metrics_summary_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}/ \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/results_nusc_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
# mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/metrics_summary_${CKPT}.json \
# ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json


#exp 5
VERSION=74
CKPT=polarbev_nmFFF

CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 4 --gpus 2\
 --vel_div mul

CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 4 --gpus 2 \
 --vel_div mul --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}/ema_23.pth -e 

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json


#exp 6
VERSION=2
CKPT=polarbev_2key_nmTFF

CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx.py --amp_backend native -b 4 --gpus 2\
--vel_div mul --norm_bbox

CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx.py -b 4 --gpus 2 \
--vel_div mul --norm_bbox --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}/ema_23.pth -e 

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json


#exp 6
VERSION=4
CKPT=polarbev_2key_nmFFF

CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx.py --amp_backend native -b 4 --gpus 2\
--vel_div mul

CUDA_VISIBLE_DEVICES=0,1 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx.py -b 4 --gpus 2 \
--vel_div mul --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}/ema_23.pth -e 

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_2key_polar_mgt0_7split_dimx/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json

