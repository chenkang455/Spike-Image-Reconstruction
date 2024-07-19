import os
from utils import *
from compare_zoo.Spk2ImgNet.nets import SpikeNet
from compare_zoo.WGSE.dwtnets import Dwt1dResnetX_TCN
from compare_zoo.SSML.model import DoubleNet
from compare_zoo.TFP.tfp_model import TFP
from compare_zoo.TFI.tfi_model import TFI
from compare_zoo.TFSTP.tfstp_model import TFSTP
# from models import SPCS_Net
from dataset import SpikeData_REDS,SpikeData_Real
from tqdm import tqdm
from metrics import compute_img_metric,compute_img_metric_single
from thop import profile
import torch.nn.functional as F
import time

# network loading
def network_construct(method_name):
    global spkrecon_net
    # CNN-based
    if method_name == 'Spk2ImgNet':
        spkrecon_net = SpikeNet(
            in_channels=13, features=64, out_channels=1, win_r=6, win_step=7
        )
        load_path = "compare_zoo/Spk2ImgNet/model_061.pth"
        load_net = True
    elif method_name == 'WGSE':
        yl_size,yh_size = 15 ,[28, 21, 18, 16, 15]
        spkrecon_net = Dwt1dResnetX_TCN(
            wvlname="db8", J=5, yl_size=yl_size, yh_size=yh_size, num_residual_blocks=3, norm=None, ks=3, store_features=True
        )
        load_path = "compare_zoo/WGSE/model_best.pt"
        load_net = True
    elif method_name.startswith("SPCS"):
        # spkrecon_net = SPCS_Net()
        # load_path = f"models/{method_name}.pth"
        load_net = True
    elif method_name == "SSML":
        spkrecon_net = DoubleNet()
        load_path = "compare_zoo/SSML/fin3g-best-lucky.pt"
        load_net = True
    # Explained
    elif method_name == "TFP":
        spkrecon_net = TFP()
        load_net = False
    elif method_name == "TFI":
        spkrecon_net = TFI()
        load_net = False
    elif method_name == "TFSTP":
        spkrecon_net = TFSTP()
        load_net = False
    spkrecon_net = spkrecon_net.cuda()
    if load_net == True:
        load_network(load_path,spkrecon_net)
    
    

# network output
def network_output(spike,method_name):
    # input
    spike_idx = len(spike[0]) // 2 
    spike = spike[:,spike_idx - 20:spike_idx + 21]
    if method_name == 'Spk2ImgNet':
        spike = torch.cat([spike,spike[:,:,-2:]],dim = 2)
    if method_name != 'TFSTP':
        spike = spike.cuda()
    # output
    if method_name.startswith('SPCS'):
        iter_num = int(method_name.split('.')[0].split('_')[-1])
        recon_img = spkrecon_net(spike,iter_num = iter_num)
    else:
        recon_img = spkrecon_net(spike)
    if method_name == 'Spk2ImgNet':
        if dataset_cls == 'REDS':
            recon_img =  torch.clamp(recon_img / 0.6, 0, 1)
        recon_img = recon_img[:,:,:250,:]
    elif method_name in ['TFP','TFI']:
        recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
    elif method_name == 'TFSTP':
        recon_img = recon_img.cuda().float()
        recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
    return recon_img


# establish metric
def metric_construct():
    global metrics
    metrics = {}
    for method_name in method_list:
        metrics[method_name] = {}  
        for metric_name in metric_list:
            metrics[method_name][metric_name] = AverageMeter()

# update metric dictionary
def metric_update(method_name,recon_img,sharp):
    for key in metric_list:
        if key in metric_pair:
            metrics[method_name][key].update(compute_img_metric(recon_img,sharp,key))
        elif key in metric_single:
            metrics[method_name][key].update(compute_img_metric_single(recon_img,key))
        else:
            ValueError("Key not found.")

# calculate metric
def metric_calculate(method_name):
    logger.info(f"Method {method_name} estimating metrics...")
    network_construct(method_name)
    for batch_idx, (spike,sharp) in enumerate(tqdm(dataloader)):
        if dataset_cls == 'Real':
            spk_length = spike.shape[1]
            for spk_idx in np.linspace(30,spk_length-30,5):
                spk_idx = int(spk_idx)
                recon_img = network_output(spike[:,spk_idx-20:spk_idx+21],method_name)
                metric_update(method_name,recon_img,sharp)
        elif dataset_cls == 'REDS':
            recon_img = network_output(spike,method_name)
            metric_update(method_name,recon_img,sharp)

# logger.info metric
def metric_print():
    for method_name in method_list:
        re_msg = method_name + '-----'
        for metric_name in metric_list:
            re_msg += metric_name + ": " + "{:.4f}".format(metrics[method_name][metric_name].avg) + "  "
        logger.info(re_msg)
        
# output img
def img_reconstruct(spike,method_name,nor = False,large = True):
    logger.info(f"Method {method_name} reconstructing img...")
    network_construct(method_name)
    recon_img = network_output(spike,method_name)
    os.makedirs(f'imgs/{spike_name}',exist_ok = True)
    recon_img = recon_img.detach().cpu()
    if nor == True:
        recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min()) 
    recon_img = recon_img * 255
    recon_img = recon_img[0,0].numpy()
    cv2.imwrite(f'imgs/{spike_name}/{method_name}.png',recon_img)
    return recon_img

# parameters estimation
def params_calculate(method_name):
    logger.info(f"Method {method_name} estimating parameters and flops...")
    network_construct(method_name)
    # input
    spike = torch.zeros((1,41,250,400)).cuda()
    spike_idx = len(spike[0]) // 2 
    spike = spike[:,spike_idx - 20:spike_idx + 21]
    if method_name == 'Spk2ImgNet':
        spike = torch.cat([spike,spike[:,:,-2:]],dim = 2)
    # output
    total = sum(p.numel() for p in spkrecon_net.parameters())
    if method_name.startswith('SPCS'):
        iter_num = int(method_name.split('.')[0].split('_')[-1])
        flops, _ = profile((spkrecon_net), inputs=(spike,iter_num))
    else:
        flops, _ = profile((spkrecon_net), inputs=(spike,))
    # test_time
    start_time = time.time()
    for _ in range(100):
        if method_name.startswith('SPCS'):
            iter_num = int(method_name.split('.')[0].split('_')[-1])
            spkrecon_net(spike,iter_num)
        else:
            spkrecon_net(spike)
    latency = (time.time() - start_time) / 100
    re_msg = (
        "Total params: %.4fM" % (total / 1e6),
        "FLOPs=" + str(flops / 1e9) + '{}'.format("G"),
        "Latency: {:.6f} seconds".format(latency)
    )    
    logger.info(re_msg)


# data load
def dataset_load(cls = 'spike'):
    global spike,dataloader,spike_name,spike_length
    if cls == 'spike':
        spike_path = 'Data/recVidarReal2019/classB/train-350kmh.dat'
        spike = load_vidar_dat(spike_path)
        spike_length = spike.shape[0]
        spike = torch.tensor(spike)[None]
        spike_name,_ = os.path.splitext(os.path.basename(spike_path))
    elif cls == 'REDS':
        dataset = SpikeData_REDS("Data/REDS","REDS",'test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    elif cls == 'Real':
        dataset = SpikeData_Real("Data/recVidarReal2019")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)

# main
def main():
    global dataloader,logger,metric_list,method_list,metric_pair,metric_single,dataset_cls 
    dataset_cls = 'spike'
    # dataset preparation
    dataset_load(cls = dataset_cls)
    logger = setup_logging('compare.log')
    # metric and method definition
    method_list = ['Spk2ImgNet','WGSE','SSML','TFP','TFI','TFSTP']
    # method_list = ['SPCS_1','SPCS_5','SPCS_10']
    
    # lower better: lpips ; higher better: psnr,ssim
    metric_pair = ['psnr','ssim','lpips']
    # lower better: niqe,brisque ; higher better: liqe_max,clipiqa
    metric_single = ['niqe','brisque','liqe_mix','clipiqa']
    
    test_metric = False 
    test_params = False 
    test_imgs = True
    test_folder_metric = False
    
    # test -- metrics
    if test_metric == True:    
        # metric_list = ['psnr','ssim','lpips','niqe','brisque','liqe_mix','clipiqa']
        metric_list = ['niqe','brisque','liqe_mix','clipiqa']

        metric_construct()
        # test -- metric
        for method_name in method_list:
            metric_calculate(method_name)
        metric_print()
        
    # test -- params
    if test_params  == True:
        for method_name in method_list:
            params_calculate(method_name)
            
    # save -- imgs
    if test_imgs == True:
        print(spike_length)
        idx = 200
        if idx < 20 or idx >= spike_length - 21:
            raise RuntimeError(f"Idx out of the length.")
        for method_name in method_list:
            img_reconstruct(spike[:,idx-20:idx+21],method_name,nor = True)
    
    # folder --test metrcis
    if test_folder_metric == True:
        folder = 'imgs/train-350kmh'
        metric_list = ['niqe','brisque','liqe_mix','clipiqa']
        metric_construct()
        for img_path in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,img_path))[:,:,0] / 255
            img = torch.tensor(img)[None,None].cuda().float()
            print(img.max(),img.min())
            method_name = img_path.split('.')[0]
            if method_name not in method_list:
                continue
            metric_update(method_name,img,None)
        metric_print()
        
if __name__ == "__main__":
    main()
