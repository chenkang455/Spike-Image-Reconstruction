<h2 align="center"> 
  <a href="">Spike-Zoo: A Comprehensive Collection of Spike-based Image Reconstruction Methods
  </a>
</h2>
<h5 align="center"> 
If you like it, please give us a star ⭐ on GitHub.  
</h5>

<h5 align="center">

[![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/chenkang455/S-SDM)
[![GitHub repo stars](https://img.shields.io/github/stars/chenkang455/Spike-Image-Reconstruction?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/chenkang455/Spike-Image-Reconstruction/stargazers)&#160;
</h5>

<p align="center">
  <img src="figures/zoo_new.png" style="width:50%;">
</p>


## 📕 Introduction 
This GitHub repository integrates various spike-based image reconstruction methods. It aims to assist in comparing the performance of previous approaches on the standard REDS dataset, Real-spike dataset, or a single spike sequence.

## 🗓️ Todo
<form>
  <label><input type="checkbox" /> Support more spike-based image reconstruction methods. (CVPR24 Zhao et al., ACM MM23 Zhu et al., TCSVT23 Zhao et al.) </label><br>
  <label><input type="checkbox" /> Support more datasets. (CVPR24 Zhao et al., TCSVT23 Zhao et al.)</label><br>
  <label><input type="checkbox" /> Support more metrics. (More non-reference metrics.)</label><br>
</form>

## 🕶 Methods
In this repository, we currently support the following methods: TFP<sup>[1]</sup>, TFI<sup>[1]</sup>, TFSTP<sup>[2]</sup>, Spk2ImgNet<sup>[3]</sup>, SSML<sup>[4]</sup>, and WGSE<sup>[5]</sup>, which take 41 spike frames as the input and reconstruct one sharp image. 

## 🔢 Datasets

## 🌏 Metrics

## 🍭 Startup

## 📞 Contact
Should you have any questions, please feel free to contact [mrchenkang@stu.pku.edu.cn](mailto:mrchenkang@stu.pku.edu.cn).

## 🙇‍ Acknowledgment
Implementations of TFP, TFI and TFSTP are from the [SpikeCV](https://spikecv.github.io/). Other methods are implemented according to the paper official repository. Implementations of non-reference metrics are from the [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch).We appreciate the effort of the contributors to these repositories.

## References
<style>
.custom-paragraph {
    line-height: 1.6; /* 调整这个值来设置你希望的行间距 */
    margin-bottom: 20px; /* 设置段落间的间距 */
}
</style>

<div class="custom-paragraph">
<p>
[1] Zhu, Lin, et al. "A retina-inspired sampling method for visual texture reconstruction." 2019 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2019.<br>
[2] Zheng, Yajing, et al. "Capture the moment: High-speed imaging with spiking cameras through short-term plasticity." IEEE Transactions on Pattern Analysis and Machine Intelligence 45.7 (2023): 8127-8142.<br>
[3] Zhao, Jing, et al. "Spk2imgnet: Learning to reconstruct dynamic scene from continuous spike stream." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.<br>
[4] Chen, Shiyan, et al. "Self-Supervised Mutual Learning for Dynamic Scene Reconstruction of Spiking Camera." IJCAI. 2022.<br>
[5] Zhang, Jiyuan, et al. "Learning temporal-ordered representation for spike streams based on discrete wavelet transforms." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 1. 2023. 
</p>
</div>


> 