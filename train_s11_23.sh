# exp 1
VERSION=0
CKPT=base_8key

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py --amp_backend native -b 4 --gpus 2 \
--key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_8key

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 4 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/lightning_logs/version_${VERSION}/ema_23.pth -e \
--key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_8key

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_8key/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_8key/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_8key/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_8key/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_8key/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



# exp 1
VERSION=0
CKPT=deformconv_mm_L1_8key_7split

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py --amp_backend native -b 4 --gpus 2 \
--deform_conv_lstm --motion_gate --convlstm_layer 1 --key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py -b 4 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}/ema_23.pth -e \
--deform_conv_lstm --motion_gate --convlstm_layer 1 --key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json





VERSION=2
CKPT=deformconv_mm_L2_8key_7split

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py --amp_backend native -b 4 --gpus 2 \
--deform_conv_lstm --motion_gate --convlstm_layer 2 --key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key

CUDA_VISIBLE_DEVICES=2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py -b 4 --gpus 2 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}/ema_23.pth -e \
--deform_conv_lstm --motion_gate --convlstm_layer 2 --key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



sleep 3600

# exp 1
VERSION=2
CKPT=base_4key_full

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py --amp_backend native -b 6 --gpus 4 \
--key 4 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_4key --full

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 6 --gpus 4 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_4key/lightning_logs/version_${VERSION}/ema_23.pth -e \
--key 4 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_4key

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_4key/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_4key/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_4key/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_4key/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_4key/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_4key/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_4key/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_4key/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_4key/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_4key/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



# exp 1
VERSION=4
CKPT=deformconv_mm_L1_4key_full

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py --amp_backend native -b 6 --gpus 4 \
--deform_conv_lstm --motion_gate --convlstm_layer 1 --key 4 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key --full

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py -b 6 --gpus 4 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/lightning_logs/version_${VERSION}/ema_23.pth -e \
--deform_conv_lstm --motion_gate --convlstm_layer 1 --key 4 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_4key/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json





# exp 1
VERSION=2
CKPT=base_8key_full

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py --amp_backend native -b 6 --gpus 4 \
--key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_8key --full

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 6 --gpus 4 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/lightning_logs/version_${VERSION}/ema_23.pth -e \
--key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_8key

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_8key/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_8key/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_8key/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_8key/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_8key/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_8key/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json



# exp 1
VERSION=4
CKPT=deformconv_mm_L1_8key_full

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py --amp_backend native -b 6 --gpus 4 \
--deform_conv_lstm --motion_gate --convlstm_layer 1 --key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key --full

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e_mm.py -b 6 --gpus 4 \
--ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}/ema_23.pth -e \
--deform_conv_lstm --motion_gate --convlstm_layer 1 --key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key

mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/results_nusc.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/metrics_summary.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/metrics_summary_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}/ \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}_${CKPT}/
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/results_nusc_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}_${CKPT}/results_nusc_${CKPT}.json
mv ./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/metrics_summary_${CKPT}.json \
./outputs/aedet_lss_r50_256x704_128x128_24e_mm_8key/lightning_logs/version_${VERSION}_${CKPT}/metrics_summary_${CKPT}.json

