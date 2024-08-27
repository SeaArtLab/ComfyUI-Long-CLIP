# ComfyUI-Long-CLIP
This project implements ComfyUI nodes for Long-CLIP, currently supporting the replacement of CLIP-L. 
For SD1.5, SDXL and Flux.1, the SeaArtLongClip module can be used to replace the original CLIP Text Encoder, expanding the token length from 77 to 248 max.
Through testing, we found that Long-CLIP improves the quality of the generated images. For Flux, Long-CLIP nicely complements the long token context inherent to T5.

## Start
In the ComfyUI/custom_nodes directory:
```
git clone https://github.com/zer0int/ComfyUI-Long-CLIP.git
```
For SDXL, 1.5 and Flux.1: Download [LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L) to models/clip -- thanks to [beichenzbc/Long-CLIP](https://github.com/beichenzbc/Long-CLIP) for making the weights available. 

-----

The original author of these nodes (for SD 1.5 and SDXL) is [https://github.com/SeaArtLab/ComfyUI-Long-CLIP](https://github.com/SeaArtLab/ComfyUI-Long-CLIP) (thanks!)

-----
## Workflow
We have specifically prepared examples for SD1.5, SDXL, and flux for your use. To simplify the demonstration, our examples are straightforward, and you do not need to install any additional plugins. This plugin also supports operations such as clip-skip.
![SD1.5](./image/SD1-5-long.png)
![SDXL](./image/SDXL-long.png)
![FLUX.1](./image/Flux.1-long.png)

