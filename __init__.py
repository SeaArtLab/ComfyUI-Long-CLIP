from . import long_clip as long_clip

NODE_CLASS_MAPPINGS = {
    "SeaArtLongClip": long_clip.SeaArtLongClip,
    "SeaArtLongXLClipMerge": long_clip.SeaArtLongXLClipMerge,
    "LongCLIPTextEncodeFlux": long_clip.LongCLIPTextEncodeFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeaArtLongClip": "SeaArtLongClip",
    "SeaArtLongXLClipMerge": "SeaArtLongXLClipMerge",
    "LongCLIPTextEncodeFlux": "LongCLIPTextEncodeFlux",
}
