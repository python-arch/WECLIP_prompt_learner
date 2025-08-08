import os, re, shutil, pathlib
from tqdm import tqdm

SRC_IMG_TRAIN = "/home/ahmedjaheen/WeCLIP/train2014"
SRC_IMG_VAL = "/home/ahmedjaheen/WeCLIP/val2014"
SRC_MASK_ROOT = "/home/ahmedjaheen/WeCLIP/coco_seg_anno"
DST_ROOT = "./MSCOCO"
ID_RE = re.compile(r"(\d{12})\.jpg$")

def prepare_split(img_dir, split_name):
    img_dest = pathlib.Path(DST_ROOT, "JPEGImages", split_name)
    mask_dest = pathlib.Path(DST_ROOT, "SegmentationClass", split_name)
    img_dest.mkdir(parents=True, exist_ok=True)
    mask_dest.mkdir(parents=True, exist_ok=True)
    list_path = pathlib.Path(DST_ROOT, f"{split_name}.txt")
    with list_path.open("w") as f:
        for img_path in tqdm(sorted(pathlib.Path(img_dir).glob("*.jpg")), desc=f"{split_name:>5}"):
            m = ID_RE.search(img_path.name)
            if not m:
                continue
            coco_id = m.group(1)
            mask_path = pathlib.Path(SRC_MASK_ROOT, f"{coco_id}.png")
            if not mask_path.exists():
                continue
            shutil.copy2(img_path, img_dest / img_path.name)
            shutil.copy2(mask_path, mask_dest / mask_path.name)
            f.write(img_path.stem + "\n")

prepare_split(SRC_IMG_TRAIN, "train")
prepare_split(SRC_IMG_VAL, "val")
print("done →", DST_ROOT)
