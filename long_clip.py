from .long_clip_model import longclip
import os
import torch
from comfy.sd import CLIP
import folder_paths
from comfy.sd1_clip import load_embed,ClipTokenWeightEncoder
from comfy.model_management import get_torch_device
from comfy import model_management
import comfy


class SDLongClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, dtype=None,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True, enable_attention_masks=False, return_projected_pooled=True, **kwargs):  # clip-vit-base-patch32
        super().__init__()

        assert layer in self.LAYERS

        self.transformer, _ = longclip.load(version, device=device)

        self.num_layers = self.transformer.transformer_layers

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.text_projection = torch.nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.clip_layer(layer_idx)
        self.layer_default = (self.layer, self.layer_idx)
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

        self.dtypes = [param.dtype for param in self.parameters()]

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def clip_layer(self, layer_idx):
        if abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_layer(self):
        self.layer = self.layer_default[0]
        self.layer_idx = self.layer_default[1]

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_options(self):
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]

    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0] - 1
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    if y == token_dict_size: #EOS token
                        y = -1
                    tokens_temp += [y]
                else:
                    if y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights += [y]
                        tokens_temp += [next_new_token]
                        next_new_token += 1
                    else:
                        print("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored", y.shape[0], current_embeds.weight.shape[1])
            while len(tokens_temp) < len(x):
                tokens_temp += [self.special_tokens["pad"]]
            out_tokens += [tokens_temp]

        n = token_dict_size
        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(next_new_token + 1, current_embeds.weight.shape[1], device=current_embeds.weight.device, dtype=current_embeds.weight.dtype)
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:-1]
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            new_embedding.weight[n] = current_embeds.weight[-1] #EOS embedding
            self.transformer.set_input_embeddings(new_embedding)

        processed_tokens = []
        for x in out_tokens:
            processed_tokens += [list(map(lambda a: n if a == -1 else a, x))] #The EOS token should always be the largest one

        return processed_tokens

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        attention_mask = None
        if self.enable_attention_masks:
            attention_mask = torch.zeros_like(tokens)
            max_token = self.transformer.get_input_embeddings().weight.shape[0] - 1
            for x in range(attention_mask.shape[0]):
                for y in range(attention_mask.shape[1]):
                    attention_mask[x, y] = 1
                    if tokens[x, y] == max_token:
                        break

        outputs = self.transformer(tokens, attention_mask, intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state)
        self.transformer.set_input_embeddings(backup_embeds)

        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        return z.float(), pooled_output

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        if "text_projection" in sd:
            self.text_projection[:] = sd.pop("text_projection")
        if "text_projection.weight" in sd:
            self.text_projection[:] = sd.pop("text_projection.weight").transpose(0, 1)
        return self.transformer.load_state_dict(sd, strict=False)
    
class SDLongTokenizer:
    def __init__(self, max_length=248, pad_with_end=True, embedding_directory=None, tokenizer_data=None, embedding_size=768, embedding_key='clip_l',  has_start_token=True, pad_to_max_length=True):
        self.tokenizer = longclip.only_tokenize ##tokenizer_class.from_pretrained(tokenizer_path)
        self.max_length = max_length
        empty = self.tokenizer('')[0]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        ##vocab = self.tokenizer.get_vocab()
        ##self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key
        self.tokenizer_data = tokenizer_data

    def _try_get_embedding(self, embedding_name:str):
        '''
        Takes a potential embedding name and tries to retrieve it.
        Returns a Tuple consisting of the embedding and any leftover string, embedding can be None.
        '''
        embed = load_embed(embedding_name, self.embedding_directory, self.embedding_size, self.embedding_key)
        if embed is None:
            stripped = embedding_name.strip(',')
            if len(stripped) < len(embedding_name):
                embed = load_embed(stripped, self.embedding_directory, self.embedding_size, self.embedding_key)
                return (embed, embedding_name[len(stripped):])
        return (embed, "")


    def tokenize_with_weights(self, text:str, return_word_ids=False):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0
        from comfy.sd1_clip import token_weights,escape_important,unescape_important

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                #if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                #parse word
                tokens.append([(t, weight) for t in self.tokenizer(word)[0][self.tokens_start:-1]])

        #reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            #determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    #break word in two and add end token
                    if is_large:
                        batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    #add end token and pad
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                    #start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t,w,i+1) for t,w in t_group])
                    t_group = []

        #fill last batch
        batch.append((self.end_token, 1.0, 0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]

        return batched_tokens


    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))

def pad_tokens(tokens,clip,add_token_num):
    if clip.pad_with_end:
        pad_token = clip.end_token
    else:
        pad_token = 0
    while add_token_num > 0:
        batch = []
        batch.append((clip.end_token, 1.0, 0))
        add_pad = clip.max_length - 1
        batch.extend([(pad_token, 1.0, 0)] * add_pad)
        tokens.append(batch)
        add_token_num -= (add_pad+1)
    return tokens

def token_num(tokens):
    n = 0
    for token in tokens:
        n += len(token)
    return n

class SDXLLongClipModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_l = None
        self.clip_g = None

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.clip_g.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_g.reset_clip_options()
        self.clip_l.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        g_tokens = g_out.shape[1]
        l_tokens = l_out.shape[1]
        min_tokens = min(g_tokens,l_tokens)
        g_out = g_out[:,:min_tokens,:]
        l_out = l_out[:,:min_tokens,:]
        return torch.cat([l_out, g_out], dim=-1), g_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        else:
            return self.clip_l.load_sd(sd)

class SDXLLongTokenizer:
    def __init__(self):
        self.clip_l = None
        self.clip_g = None

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        g_tokens = token_num(out["g"])
        l_tokens = token_num(out["l"])
        if g_tokens > l_tokens:
            out["l"] = pad_tokens(out["l"],self.clip_l,g_tokens-l_tokens)
        elif l_tokens > g_tokens:
            out["g"] = pad_tokens(out["g"],self.clip_g,l_tokens-g_tokens)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

class LongCLIPFluxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_l = None
        self.t5xxl = None

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]

        # Encode using Long-CLIP
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        # Encode using T5XXL
        t5_out, t5_pooled = self.t5xxl.encode_token_weights(token_weight_pairs_t5)

        return t5_out, l_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)

class LongCLIPFluxTokenizer:
    def __init__(self):
        self.clip_l = None
        self.t5xxl = None

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        # Tokenize with both Long-CLIP and T5XXL
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)  # Long-CLIP tokenization
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids)  # T5XXL tokenization

        # Check the number of tokens
        l_tokens = token_num(out["l"])
        t5_tokens = token_num(out["t5xxl"])

        # Leaving this here as a reminder: Do NOT pad T5XXL!
        if l_tokens > t5_tokens:
            pass  # Do not pad T5XXL

        return out

    def untokenize(self, token_weight_pair):
        # Untokenize using Long-CLIP tokenizer
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}

class SeaArtLongXLClipMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "clip_name": (folder_paths.get_filename_list("clip"), ),
                              "clip": ("CLIP", ),
                             }}

    CATEGORY = "SeaArt"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "do"

    def do(self, clip_name, clip):
        clip_clone = clip.clone()
        clip_path = folder_paths.get_full_path("clip", clip_name)
        load_device = model_management.text_encoder_device()
        device = model_management.text_encoder_offload_device()
        dtype = model_management.text_encoder_dtype(load_device)
        clip_l = SDLongClipModel(version=clip_path,layer="hidden", layer_idx=-2, device=device, dtype=dtype, layer_norm_hidden_state=False)
        sdxl_long_clip_model = SDXLLongClipModel()
        sdxl_long_clip_model.clip_l = clip_l
        sdxl_long_clip_model.clip_g = clip_clone.cond_stage_model.clip_g
        clip_clone.cond_stage_model = sdxl_long_clip_model
        embedding_directory = folder_paths.get_folder_paths("embeddings")
        long_tokenizer = SDXLLongTokenizer()
        tokenizer_clip_l = SDLongTokenizer(embedding_directory=embedding_directory)
        long_tokenizer.clip_l = tokenizer_clip_l
        long_tokenizer.clip_g = clip_clone.tokenizer.clip_g
        clip_clone.tokenizer = long_tokenizer
        return (clip_clone,)

class SeaArtLongClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "clip_name": (folder_paths.get_filename_list("clip"), ),
                             }}

    CATEGORY = "SeaArt"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "do"

    def do(self, clip_name):
        class EmptyClass:
            pass
        clip_target = EmptyClass()
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_target.params = {"version":clip_path}
        clip_target.clip = SDLongClipModel
        clip_target.tokenizer = SDLongTokenizer
        embedding_directory = folder_paths.get_folder_paths("embeddings")
        clip = CLIP(clip_target, embedding_directory=embedding_directory)
        return (clip,)
    
class LongCLIPTextEncodeFlux:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip_name": (folder_paths.get_filename_list("clip"), ),
            "clip": ("CLIP", ),
        }}

    CATEGORY = "SeaArt"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "do"

    def do(self, clip_name, clip):
        clip_clone = clip.clone()
        clip_path = folder_paths.get_full_path("clip", clip_name)
        load_device = model_management.text_encoder_device()
        device = model_management.text_encoder_offload_device()
        dtype = model_management.text_encoder_dtype(load_device)
        longclip_model = SDLongClipModel(version=clip_path, layer="hidden", layer_idx=-2, device=device, dtype=dtype, max_length=248)
        flux_clip_model = LongCLIPFluxModel()
        flux_clip_model.clip_l = longclip_model
        flux_clip_model.t5xxl = clip_clone.cond_stage_model.t5xxl
        clip_clone.cond_stage_model = flux_clip_model
        long_tokenizer = LongCLIPFluxTokenizer()
        long_tokenizer.clip_l = SDLongTokenizer(embedding_directory=clip_clone.tokenizer.clip_l.embedding_directory, max_length=248)
        long_tokenizer.t5xxl = clip_clone.tokenizer.t5xxl
        clip_clone.tokenizer = long_tokenizer
        return (clip_clone,)
