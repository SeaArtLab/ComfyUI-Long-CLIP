from . import long_clip as long_clip

NODE_CLASS_MAPPINGS = {
    "SeaArtLongClip": long_clip.SeaArtLongClip,
    "SeaArtLongXLClipMerge": long_clip.SeaArtLongXLClipMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeaArtLongClip": "SeaArtLongClip",
    "SeaArtLongXLClipMerge": "SeaArtLongXLClipMerge",
}