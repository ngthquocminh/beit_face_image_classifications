from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch
from timm.models import create_model
from argparse import Namespace
args = Namespace()
import torch
from timm.models import create_model
from collections import OrderedDict
import utils
from scipy import interpolate
import numpy as np

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(0, './unilm/beit2')
import modeling_pretrain


args.model = 'beit_base_patch16_224'
args.input_size = 224
args.nb_classes = 2
args.finetune = 'mp_rank_00_model_states.pt'
model = create_model(
    args.model,
    pretrained=False,
    num_classes=args.nb_classes,
    drop_rate=0,
    drop_path_rate=0,
    attn_drop_rate=0,
    drop_block_rate=None,
    use_mean_pooling=True,
    init_scale=0,
    use_rel_pos_bias=True,
    use_abs_pos_emb=False,
    init_values=0.1,
    qkv_bias=True,
)

patch_size = model.patch_embed.patch_size
print("Patch size = %s" % str(patch_size))
args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
args.patch_size = patch_size

if args.finetune:
    if args.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.finetune, map_location='cpu')

    print("Load ckpt from %s" % args.finetune)
    checkpoint_model = None
    for model_key in "model|module".split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    if (checkpoint_model is not None) and ("" != ''):
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                pass
        checkpoint_model = new_dict

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            if args.robust_test == 'imagenet_r':
                mask = torch.tensor(imagenet_a_r_indices.imagenet_r_mask)
                checkpoint_model[k] = checkpoint_model[k][mask]
            elif args.robust_test == 'imagenet_a':
                mask = torch.tensor(imagenet_a_r_indices.imagenet_a_mask)
                checkpoint_model[k] = checkpoint_model[k][mask]
            else:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

    if getattr(model, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        print("Expand the shared relative position embedding to each transformer block. ")
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                print("Position interpolate for %s from %dx%d to %dx%d" % (
                    key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                print("Original positions = %s" % str(x))
                print("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias
    # interpolate position embedding
    if ('pos_embed' in checkpoint_model) and (model.pos_embed is not None):
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    # utils.load_state_dict(model, checkpoint_model, prefix="")
    model.load_state_dict(checkpoint_model, strict=False)


class InferModel(nn.Module):
    def __init__(self,model_s):
        super().__init__()
        self.model_s = model_s.eval()

    def forward(self, image):

        images = image.unsqueeze(0) 
        with torch.no_grad():
            logits_s = self.model_s(images)
            probs = F.softmax(logits_s, dim=-1)
        return probs[0].cpu().detach().numpy()

infer_model = InferModel(model)

def classify_frame(frame):
    # Áp dụng các biến đổi cho hình ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(frame)

    # Phân loại bằng mô hình
    prob = infer_model(image_tensor)
    label = 'real' if np.argmax(prob) == 1 else 'fake'
    print(label)
    return label
# Đường dẫn đến tấm hình cần tải
# image_path = "test1/4.png"

# # Đọc hình ảnh
# image = Image.open(image_path).convert("RGB")

# # Áp dụng các biến đổi (nếu cần)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # Áp dụng các biến đổi cho hình ảnh
# image_tensor = transform(image)

# # Thêm một chiều cho batch (vì model thường đầu vào là batch)
# # image_tensor = image_tensor.unsqueeze(0)

# prob = infer_model(image_tensor)
# label = ''
# predict = np.argmax(prob)
# print(predict)


