# test parameters flops and latency
CUDA_VISIBLE_DEVICES=0 python compare.py \
--test_params \
--save_name 'logs/params.log' \
--methods 'Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP'

# test metric on the REDS dataset
CUDA_VISIBLE_DEVICES=0 python compare.py \
--test_metric \
--save_name 'logs/reds_metric.log' \
--methods 'Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP' \
--cls 'REDS' \
--metrics 'psnr,ssim,lpips,niqe,brisque,liqe_mix,clipiqa'

# test metric on the real-spike dataset
CUDA_VISIBLE_DEVICES=0 python compare.py \
--test_metric \
--save_name 'logs/real_metric.log' \
--methods 'Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP' \
--cls 'Real' \
--metrics 'niqe,brisque,liqe_mix,clipiqa'

# visualize spike 
CUDA_VISIBLE_DEVICES=0 python compare.py \
--test_imgs \
--methods 'Spk2ImgNet,WGSE,SSML,TFP,TFI,TFSTP' \
--cls 'spike' \

