<h2 align="center"> 
  <a href="">Spike-based Image Reconstruction ZOO</a>
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

## 🕶 Methods
In this repository, we currently support the following methods: TFP<sup>[1]</sup>, TFI<sup>[1]</sup>, TFSTP<sup>[2]</sup>, Spk2ImgNet<sup>[3]</sup>, SSML<sup>[4]</sup>, and WGSE<sup>[5]</sup>, which take 41 spike frames as the input and reconstruct one sharp image.

## 📞 Contact
Should you have any questions, please feel free to contact [mrchenkang@stu.pku.edu.cn](mailto:mrchenkang@stu.pku.edu.cn).

## 🙇‍ Acknowledgment
The implementations of TFP, TFI and TFSTP are from the [SpikeCV](https://spikecv.github.io/). Other methods are implemented according to the paper official repository. We appreciate the effort of the contributors to these repositories.

## References
> [1] Zhu, Lin, et al. "A retina-inspired sampling method for visual texture reconstruction." 2019 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2019.
> 
> [2] Zheng, Yajing, et al. "Capture the moment: High-speed imaging with spiking cameras through short-term plasticity." IEEE Transactions on Pattern Analysis and Machine Intelligence 45.7 (2023): 8127-8142.
> 
> [3] Zhao, Jing, et al. "Spk2imgnet: Learning to reconstruct dynamic scene from continuous spike stream." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
> 
> [4] Chen, Shiyan, et al. "Self-Supervised Mutual Learning for Dynamic Scene Reconstruction of Spiking Camera." IJCAI. 2022.
> 
> [5] Zhang, Jiyuan, et al. "Learning temporal-ordered representation for spike streams based on discrete wavelet transforms." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 1. 2023. 