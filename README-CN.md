# ComfyUI-Long-CLIP
本项目是long-clip的comfyui实现，目前支持clip-l的替换，对于SD1.5可以使用SeaArtLongClip模块加载后替换模型中原本的clip，token的长度由77扩大至248，经过测试我们发现long-clip对成图质量有提升作用，对于SDXL模型由于clip-g的clip-long模型没有出现，所以我们的处理流程如下，对于较小的token按照原本max_len的整数倍扩大，由于最后多出来的为pad_token，所以对多余部分进行了裁剪，由于SDXL中clip-g的特征占比更多，您可能看见的更多是画面出现更多细节，最后喜欢本项目请帮我们点赞

## Start
```
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/custom_nodes
git clone git@github.com:SeaArtLab/ComfyUI-Long-CLIP.git
```
下载[LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L)到 models/checkpoints，同时感谢[Long-CLIP](https://github.com/beichenzbc/Long-CLIP/tree/main)开放权重，后续LongCLIP-G权重开放后，我们会同步支持！

## Workflow
我们特意准备了SD1.5和SDXL的使用例子，为了简化演示，我们例子简单，您无需安装其他插件
![SD1.5](./image/SD1-5-long.png)
![SDXL](./image/SDXL-long.png)