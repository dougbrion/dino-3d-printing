import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

DIM = 256
HALF_DIM = int(DIM / 2)

ROOT_DIR = "/home/cam/Documents/FastData"
CSV_PATH = os.path.join(ROOT_DIR, "dataset2/final_all_filtered_pixels_nodark_even.csv")

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        x = row["nozzle_tip_x"]
        y = row["nozzle_tip_y"]

        img_path = os.path.join(ROOT_DIR, row["img_path"])
        img = Image.open(img_path)

        left = x - HALF_DIM
        right = x + HALF_DIM
        upper = y
        lower = y + DIM
        img = img.crop((left, upper, right, lower))

        new_img_path = img_path.replace("dataset2", "dino-test")
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        
        img.save(new_img_path)
