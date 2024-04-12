# BEV Exp 1
VERSION=0
CKPT=n_m_t_t_t_bc_p1_b1_bd


CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1 \
--bev_detach

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1 \
--bev_detach

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json





# BEV Exp 2
VERSION=1
CKPT=n_m_t_t_t_bc_b1_pd

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_bev_1x1 \
# --pers_detach

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_bev_1x1 \
--pers_detach

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



# BEV Exp 3
VERSION=2
CKPT=n_m_t_t_t_bc_p1_b1_pd_bd

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_pers_1x1 \
# --to_bev_1x1 \
# --pers_detach \
# --bev_detach 

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1 \
--pers_detach \
--bev_detach

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



# BEV Exp 4
VERSION=3
CKPT=n_m_t_t_t_bc_p1_pd_b1_bd_bidir

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear
# --to_pers_1x1 \
# --to_bev_1x1 \
# --pers_detach \
# --bev_detach
# --consis_bidirec

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear
--to_pers_1x1 \
--to_bev_1x1 \
--pers_detach \
--bev_detach
--consis_bidirec

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



# BEV Exp 5
VERSION=4
CKPT=n_m_t_t_t_bc_p1_b1

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_pers_1x1 \
# --to_bev_1x1

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



# BEV Exp 6
VERSION=5
CKPT=n_m_t_t_t_bc_b1

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_bev_1x1

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_bev_1x1

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json


# BEV Exp 7
VERSION=6
CKPT=n_m_t_t_t_bc_p1_b1_pd_bd_125

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_pers_1x1 \
# --to_bev_1x1 \
# --pers_detach \
# --bev_detach \
# --consis_loss_weight 0.125

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1 \
--pers_detach \
--bev_detach \
--consis_loss_weight 0.125

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json




# BEV Exp 8
VERSION=7
CKPT=n_m_t_t_t_bc_p1_b1_125

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_pers_1x1 \
# --to_bev_1x1 \
# --consis_loss_weight 0.125

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1 \
--consis_loss_weight 0.125

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json


# BEV Exp 9
VERSION=8
CKPT=n_m_t_t_t_bc_p1_b1_pd_bd_5

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_pers_1x1 \
# --to_bev_1x1 \
# --pers_detach \
# --bev_detach \
# --consis_loss_weight 0.5

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1 \
--pers_detach \
--bev_detach \
--consis_loss_weight 0.5

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json




# BEV Exp 10
VERSION=9
CKPT=n_m_t_t_t_bc_p1_b1_5

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_pers_1x1 \
# --to_bev_1x1 \
# --consis_loss_weight 0.5

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1 \
--consis_loss_weight 0.5

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json




# BEV Exp 11
VERSION=10
CKPT=n_m_t_t_t_bc_p1_b1_pd_bd_1

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_pers_1x1 \
# --to_bev_1x1 \
# --pers_detach \
# --bev_detach \
# --consis_loss_weight 1

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1 \
--pers_detach \
--bev_detach \
--consis_loss_weight 1

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json




# BEV Exp 12
VERSION=11
CKPT=n_m_t_t_t_bc_p1_b1_1

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_pers_1x1 \
# --to_bev_1x1 \
# --consis_loss_weight 1

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1 \
--consis_loss_weight 1

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json






# BEV Exp 13
VERSION=12
CKPT=n_m_t_t_t_bc_p1_b1_pd_bd_pick

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_pers_1x1 \
# --to_bev_1x1 \
# --pers_detach \
# --bev_detach

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1 \
--pers_detach \
--bev_detach

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json




# BEV Exp 14
VERSION=13
CKPT=n_m_t_t_t_bc_p1_b1_pick

# CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py --amp_backend native -b 8 --gpus 2 \
# --vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
# --neck_consistency \
# --neck_consistency_type bilinear \
# --to_pers_1x1 \
# --to_bev_1x1

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx.py -b 8 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ema_23.pth -e \
--vel_div mul --norm_bbox --azimuth_center --radius_center  --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck --root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck \
--neck_consistency \
--neck_consistency_type bilinear \
--to_pers_1x1 \
--to_bev_1x1

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_polar_mgt0_7split_dimx_neck/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json

