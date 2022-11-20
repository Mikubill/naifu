import argparse
import os.path
import re
import tempfile
import zipfile
from pathlib import Path

import deepdanbooru as dd
import numpy as np
import tensorflow as tf
from PIL import Image
from basicsr.utils.download_util import load_file_from_url
from tqdm import tqdm

re_special = re.compile(r"([\\()])")


def get_deepbooru_tags_model(model_path=None):
    if model_path is None:
        model_path = os.path.abspath(os.path.join(tempfile.gettempdir(), "deepbooru"))
    if not os.path.exists(os.path.join(model_path, "project.json")):
        # there is no point importing these every time

        load_file_from_url(
            r"https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip",
            model_path,
        )
        with zipfile.ZipFile(
                os.path.join(model_path, "deepdanbooru-v3-20211112-sgd-e28.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(model_path)
        os.remove(os.path.join(model_path, "deepdanbooru-v3-20211112-sgd-e28.zip"))

    tags = dd.project.load_tags_from_project(model_path)
    model = dd.project.load_model_from_project(model_path, compile_model=False)
    return model, tags


def get_deepbooru_tags_from_model(
        model,
        tags,
        pil_image,
        threshold,
        alpha_sort=False,
        use_spaces=True,
        use_escape=True,
        include_ranks=False,
):
    width = model.input_shape[2]
    height = model.input_shape[1]
    image = np.array(pil_image)
    image = tf.image.resize(
        image,
        size=(height, width),
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=True,
    )
    image = image.numpy()  # EagerTensor to np.array
    image = dd.image.transform_and_pad_image(image, width, height)
    image = image / 255.0
    image_shape = image.shape
    image = image.reshape((1, image_shape[0], image_shape[1], image_shape[2]))

    y = model.predict(image)[0]

    result_dict = {}

    for i, tag in enumerate(tags):
        result_dict[tag] = y[i]

    unsorted_tags_in_theshold = []
    result_tags_print = []
    for tag in tags:
        if result_dict[tag] >= threshold:
            if tag.startswith("rating:"):
                continue
            unsorted_tags_in_theshold.append((result_dict[tag], tag))
            result_tags_print.append(f"{result_dict[tag]} {tag}")

    # sort tags
    result_tags_out = []
    sort_ndx = 0
    if alpha_sort:
        sort_ndx = 1

    # sort by reverse by likelihood and normal for alpha, and format tag text as requested
    unsorted_tags_in_theshold.sort(key=lambda y: y[sort_ndx], reverse=True)
    for weight, tag in unsorted_tags_in_theshold:
        tag_outformat = tag
        if use_spaces:
            tag_outformat = tag_outformat.replace("_", " ")
        if use_escape:
            tag_outformat = re.sub(re_special, r"\\\1", tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{weight:.3f})"

        result_tags_out.append(tag_outformat)

    return ", ".join(result_tags_out)


def main(args):
    EXTS = ['*.jpg', '*.png', '*.jpeg', '*.gif', '*.webp', '*.bmp']

    files_grabbed = []
    for exts in EXTS:
        files_grabbed.extend(Path(args.path).glob(exts))

    print("Loading model...")
    model, tags = get_deepbooru_tags_model(args.model_path)
    print("... Loaded.")

    for image_path in tqdm(files_grabbed, desc="Processing"):
        image = Image.open(image_path).convert("RGB")
        prompt = get_deepbooru_tags_from_model(
            model,
            tags,
            image,
            args.threshold,
            alpha_sort=args.alpha_sort,
            use_spaces=args.use_spaces,
            use_escape=args.use_escape,
            include_ranks=args.include_ranks,
        )
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_filename = os.path.join(args.path, f"{image_name}.txt")
        print(f"[*] {txt_filename} -> {prompt}")
        with open(txt_filename, 'w') as f:
            f.write(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--threshold", type=int, default=0.75)
    parser.add_argument("--use-spaces", type=bool, default=True)
    parser.add_argument("--use-escape", type=bool, default=False)
    parser.add_argument("--include-ranks", type=bool, default=False)
    parser.add_argument("--alpha-sort", type=bool, default=False)

    main(parser.parse_args())