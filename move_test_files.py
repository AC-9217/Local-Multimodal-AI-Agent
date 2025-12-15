import shutil
from pathlib import Path

def move_files():
    # Move image
    src_img = Path("test_image.jpg")
    dst_img_dir = Path("data/images")
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    if src_img.exists():
        shutil.move(str(src_img), str(dst_img_dir / src_img.name))
        print(f"Moved {src_img} to {dst_img_dir}")
    else:
        print(f"{src_img} not found in root")

if __name__ == "__main__":
    move_files()
