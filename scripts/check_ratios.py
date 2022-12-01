import argparse
from pathlib import Path
from PIL import Image

def gen_buckets(base_res=(512, 512), max_size=512 * 768, dim_range=(256, 1024), divisor=64):
    min_dim, max_dim = dim_range
    buckets = set()

    w = min_dim
    while w * min_dim <= max_size and w <= max_dim:
        h = min_dim
        got_base = False
        while w * (h + divisor) <= max_size and (h + divisor) <= max_dim:
            if w == base_res[0] and h == base_res[1]:
                got_base = True
            h += divisor
        if (w != base_res[0] or h != base_res[1]) and got_base:
            buckets.add(base_res)
        buckets.add((w, h))
        w += divisor

    h = min_dim
    while h / min_dim <= max_size and h <= max_dim:
        w = min_dim
        while h * (w + divisor) <= max_size and (w + divisor) <= max_dim:
            w += divisor
        buckets.add((w, h))
        h += divisor

    return sorted(buckets, key=lambda sz: sz[0] * 4096 - sz[1])

def arb_transform(source_size, size):
    x, y = source_size
    short, long = (x, y) if x <= y else (y, x)

    w, h = size
    min_crop, max_crop = (w, h) if w <= h else (h, w)
    ratio_src, ratio_dst = long / short, max_crop / min_crop

    if ratio_src > ratio_dst:
        new_w, new_h = ((min_crop, int(min_crop * ratio_src)) if x < y else (int(min_crop * ratio_src), min_crop))
    elif ratio_src < ratio_dst:
        new_w, new_h = ((max_crop, int(max_crop / ratio_src)) if x > y else (int(max_crop / ratio_src), max_crop))
    else:
        new_w, new_h = w, h

    return new_w, new_h

def build_ratio_counter(x, buckets, show_path=False):
    b = sorted(buckets.keys())
    
    def closest_bucket(x, y):
        return b[min(range(len(b)), key=lambda i: abs(b[i] - x / y),)]
    
    arb_counter = dict()
    for img_path in Path(x).glob("*.*"):
        if img_path.suffix not in [".jpg", ".png", ".bmp"]:
            continue
        img = Image.open(img_path)
        x, y = img.size
        ratio = closest_bucket(x, y)
        arb_counter[ratio] = 1 if ratio not in arb_counter else arb_counter[ratio]+1
        if show_path:
            print(f"{img_path} ({x}, {y}) -> {buckets[ratio]}")
        
    return arb_counter

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", default=None, type=str, required=True, help="Path to instance images."
    )
    parser.add_argument(
        "-c", default=None, type=str, required=False, help="Path to class images."
    )
    parser.add_argument(
        "--show-path",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    buckets = dict()
    aspects = gen_buckets()
    for x, y in aspects:
        buckets[x / y] = (x, y)
    
    i_ratios = build_ratio_counter(args.i, buckets, args.show_path)
    print(f"Summary: {args.i}")
    for k, v in i_ratios.items(): print(f"{buckets[k]}: {v} images") 
    
    if args.c:
        c_ratios = build_ratio_counter(args.c, buckets, args.show_path)
        print(f"Summary: {args.c}")
        for k, v in c_ratios.items(): print(f"{buckets[k]}: {v} images")
        
        print("transforms:")
        i_len = len(list(Path(args.i).glob("*.*")))
        c_len = len(list(Path(args.c).glob("*.*")))
        base = i_ratios if i_len > c_len else c_ratios
        to = c_ratios if base == i_ratios else i_ratios
        for k, _ in base.items():
            if k not in to:
                st = [(abs(k - s), s) for s in to.keys()]
                st.sort(key=lambda x: x)
                cloest = st[0][1]
                intermediate = arb_transform(buckets[k], buckets[cloest])
                print(f"{buckets[k]} -> {intermediate} -> {buckets[cloest]}")

if __name__ == "__main__":
    main()
