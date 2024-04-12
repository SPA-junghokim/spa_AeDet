
for CKPT in 0 1 2 3 4 5 6 7 8 9
do

#   python scripts/visualize.py $CKPT outputs/aedet_lss_r50_256x704_128x128_100e_polar_mgt0_100data/results_nusc.json outputs/aedet_lss_r50_256x704_128x128_100e_polar_mgt0_100data/visual

#python scripts/visualize.py $CKPT outputs/aedet_lss_r50_256x704_128x128_100e_polar_mgt0_100data2/results_nusc.json ./visual_result/overfit data/nuScenes/nuscenes_12hz_infos_train_100data.pkl
python scripts/visualize.py $CKPT outputs/aedet_lss_r50_256x704_128x128_100e_100data/results_nusc.json ./visual_result/aedet data/nuScenes/nuscenes_12hz_infos_train_20data.pkl 0.2
python scripts/visualize.py $CKPT outputs/aedet_lss_r50_256x704_128x128_100e_polar_mgt0_100data/results_nusc.json ./visual_result/polar data/nuScenes/nuscenes_12hz_infos_train_20data.pkl 0.2

done

