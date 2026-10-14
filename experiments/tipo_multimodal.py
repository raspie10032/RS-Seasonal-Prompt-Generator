"""Private image-conditioned supervision; pixels are never replaced with captions.

Use Gemma's image processor with the same media placement as ComfyUI. Text rows
retain the existing contract. Visual annotations describe retained observations;
the authored answer may change the scene according to the Korean instruction.
"""
import hashlib
from pathlib import Path

from tipo_prompt import VERSION, build_user_prompt

IMAGE_CONTRACT = 'comfy-gemma4-image/1'


def validate_image_row(row):
    if row.get('prompt_contract') != VERSION or row.get('purpose') != 'training':
        raise ValueError('Image training requires the versioned training contract')
    path = Path(row['input']['image'])
    if not path.is_absolute() or path.resolve() != Path(row['source']['image']).resolve():
        raise ValueError('Image input must match the local provenance image')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['source']['image_sha256']:
        raise ValueError('Image bytes differ from reviewed source')
    review = row.get('visual_review', {})
    if (review.get('image_sha256') != row['source']['image_sha256']
            or not review.get('reviewer') or not review.get('observed_tags')
            or review.get('style') not in ('2d', '3d')):
        raise ValueError('Hash-bound visual review with observed tags/style is required')
    retained = row.get('visual_required', [])
    if not retained or not set(retained) <= set(review['observed_tags']):
        raise ValueError('Retained visual labels must be reviewed observations')
    if not set(retained) <= set(row['required']):
        raise ValueError('Retained visual labels must be scored')
    if 'messages' in row:
        raise ValueError('Rebuild multimodal messages from input; no stored text substitute')


def make_processor(base, max_soft_tokens=280):
    # Local checkpoint has a tokenizer but no processor config. Explicitly use
    # the installed reference classes and record their settings/version in run.
    from transformers import (AutoTokenizer, Gemma4Processor, Gemma4ImageProcessor,
                              Gemma4AudioFeatureExtractor, Gemma4VideoProcessor)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    return Gemma4Processor(
        Gemma4AudioFeatureExtractor(), Gemma4ImageProcessor(max_soft_tokens=max_soft_tokens),
        tokenizer, Gemma4VideoProcessor(), chat_template=tokenizer.chat_template,
        image_seq_length=max_soft_tokens)


def image_prompt(processor, row, answer=None):
    inputs = row['input']
    text = build_user_prompt(inputs['prompt'], inputs.get('tag_length', 'long'),
                             inputs.get('ban_tags', ''), True)
    # Processor expands this ONE placeholder into begin/image tokens/end.
    # The native node places media after text, with two newlines on each side.
    text += '\n\n' + processor.tokenizer.image_token + '\n\n'
    messages = [{'role': 'user', 'content': text}]
    if answer is not None:
        messages.append({'role': 'assistant', 'content': answer})
    rendered = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=answer is None, enable_thinking=False)
    # HF's template trims trailing user whitespace. ComfyUI retains these two
    # newlines after the image, so restore them before processor expansion.
    return rendered.replace(processor.tokenizer.image_token + '<turn|>',
                            processor.tokenizer.image_token + '\n\n<turn|>', 1)


def image_inputs(processor, row, answer=None):
    from PIL import Image, ImageOps
    import numpy as np
    import torch
    from torchvision.transforms import functional as tvf
    from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
    with Image.open(row['input']['image']) as image:
        image = ImageOps.exif_transpose(image)
        # Require explicit prepared RGB for alpha-bearing sources, as in eval.
        if 'A' in image.getbands() or 'transparency' in image.info:
            raise ValueError('Prepare and hash an explicit RGB composite before training')
        rgb = image.convert('RGB')
    # Match LoadImage -> Comfy tokenizer's uint8 bicubic path. HF's generic
    # backend resizes a float copy and has slightly different pixel values.
    samples = torch.from_numpy(np.array(rgb).copy()).permute(2, 0, 1).float()/255
    h, w = samples.shape[-2:]
    settings = processor.image_processor
    size = get_aspect_ratio_preserving_size(h, w, settings.patch_size,
        settings.max_soft_tokens * settings.pooling_kernel_size**2, settings.pooling_kernel_size)
    pixels = (samples.clamp(0, 1)*255).to(torch.uint8)
    if tuple(size) != (h, w):
        pixels = tvf.resize(pixels, list(size), interpolation=tvf.InterpolationMode.BICUBIC, antialias=True)
    pixels = pixels.float()*(1.0/255.0)
    return processor(text=image_prompt(processor, row, answer), images=pixels,
                     do_resize=False, do_rescale=False, return_tensors='pt', add_special_tokens=False)


def encode_multimodal(processor, row, max_length):
    from .train_gemma4_tipo import encode_example
    if not row['input'].get('image'):
        result = encode_example(processor.tokenizer, row, max_length)
        result['model_inputs'] = None
        return result
    import torch
    prompt = image_inputs(processor, row)
    whole = image_inputs(processor, row, row['answer'])
    pids, ids = prompt['input_ids'][0], whole['input_ids'][0]
    if not torch.equal(ids[:len(pids)], pids):
        raise ValueError('Multimodal assistant prefix changed')
    if len(ids) > max_length or len(ids) <= len(pids):
        raise ValueError(f'Invalid multimodal sequence length: {len(ids)} (limit {max_length})')
    if not torch.any(pids == processor.tokenizer.image_token_id):
        raise ValueError('Missing image tokens; refusing text-only downgrade')
    if 'pixel_values' not in whole or 'image_position_ids' not in whole:
        raise ValueError('Missing image pixels/positions')
    return {'input_ids': ids.tolist(), 'labels': [-100]*len(pids) + ids[len(pids):].tolist(),
            'prompt_length': len(pids), 'model_inputs': dict(whole)}
