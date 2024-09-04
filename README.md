## Update 31/AUG/2024
- Added support for safetensors
- You can download my improved, fine-tuned Long-CLIP models [here on HF 🤗](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14/tree/main).
- ⚠️ Note: Have an `issue`? I know you can't open an `issue on a Fork`. Open an issue [with the original repo](https://github.com/SeaArtLab/ComfyUI-Long-CLIP/issues) instead.
- I am watching 'all activity' on the original repo, so I'll see it. So if you have a request or complaint, please do tell.
- If you try to DM me on X, you're apparently more likely than not to be flagged as a spambot, and I won't get notified. Happened with the .safetensors request I only saw by fat chance due to checking another DM 😔. Sorry! So yeah, boldly go ahead and open an issue on the original SeaArtLab, and I'll see it. (If I had known this was a 'takeover' with no response to the pull, I wouldn't have forked it, oops!)

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

