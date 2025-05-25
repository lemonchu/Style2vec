import os
import random
import shutil

font_ds_dir = 'font_ds/fonts'
train_dir = os.path.join(font_ds_dir, 'train')
test_dir = os.path.join(font_ds_dir, 'test')

# Ensure that the train and test directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all ttf files
ttf_files = [f for f in os.listdir(font_ds_dir) if f.endswith('.ttf')]

# Shuffle the file list randomly
random.shuffle(ttf_files)

print(len(ttf_files))

# Ensure there are at least 512 ttf files
if len(ttf_files) < 512:
    raise ValueError("There are fewer than 512 ttf files. Please ensure there are enough ttf files.")

# Distribute files into train and test folders
train_files = ttf_files[:512]
test_files = ttf_files[512:]

# Move files to the train directory
for file in train_files:
    src_path = os.path.join(font_ds_dir, file)
    dst_path = os.path.join(train_dir, file)
    shutil.move(src_path, dst_path)

# Move files to the test directory
for file in test_files:
    src_path = os.path.join(font_ds_dir, file)
    dst_path = os.path.join(test_dir, file)
    shutil.move(src_path, dst_path)

print(f"Moved {len(train_files)} ttf files to {train_dir}")
print(f"Moved {len(test_files)} ttf files to {test_dir}")