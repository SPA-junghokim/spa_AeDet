

for CKPT in 22 21 20 19 18 17 16 15 14 13 12 
do

    python ./exps/aedet/aedet_lss_r50_256x704_128x128_24e.py -b 1 --gpus 1 \
    --ckpt_path ./outputs/aedet_lss_r50_256x704_128x128_24e_4key/baseline_7split/ema_${CKPT}.pth -e \
    --key 8 --default_root_dir ./outputs/aedet_lss_r50_256x704_128x128_24e_4key

    n=${#CKPT}
    if [ $n -eq 1 ]
    then 
        CKPT="0${CKPT}"
    fi
	
    mv outputs/aedet_lss_r50_256x704_128x128_24e_4key/metrics_summary.json \
    outputs/aedet_lss_r50_256x704_128x128_24e_4key/metrics_summary$CKPT.json

done



