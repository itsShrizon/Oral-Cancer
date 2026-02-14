"""Check if there's data leakage through image similarity or file names"""
from data.dataset import load_dataset2_split
import os

print("Analyzing Dataset 2 for potential leakage...")
print("="*70)

# Load all splits
train_p, _, _ = load_dataset2_split('train')
val_p, _, _ = load_dataset2_split('val')
test_p, _, _ = load_dataset2_split('test')

print(f"\nDataset 2 Split Sizes:")
print(f"  Train: {len(train_p)}")
print(f"  Val:   {len(val_p)}")
print(f"  Test:  {len(test_p)}")

# Get base filenames
train_files = set([os.path.basename(p) for p in train_p])
val_files = set([os.path.basename(p) for p in val_p])
test_files = set([os.path.basename(p) for p in test_p])

print(f"\nChecking for duplicate filenames across splits...")
train_val_overlap = train_files & val_files
train_test_overlap = train_files & test_files
val_test_overlap = val_files & test_files

print(f"  Train-Val overlap: {len(train_val_overlap)} files")
print(f"  Train-Test overlap: {len(train_test_overlap)} files")
print(f"  Val-Test overlap: {len(val_test_overlap)} files")

if train_test_overlap:
    print(f"\n⚠️  WARNING: Found {len(train_test_overlap)} duplicate filenames in train and test!")
    print("Sample duplicates:", list(train_test_overlap)[:5])

# Check source folders
print(f"\nAnalyzing source folder distribution in TEST set:")
from collections import Counter
test_sources = Counter()
for p in test_p:
    if 'Training' in p:
        test_sources['Training'] += 1
    elif 'Validation' in p:
        test_sources['Validation'] += 1

print(f"  From Training folder: {test_sources['Training']}")
print(f"  From Validation folder: {test_sources['Validation']}")
print(f"  Ratio: {test_sources['Training']/len(test_p)*100:.1f}% / {test_sources['Validation']/len(test_p)*100:.1f}%")

# Check class distribution
print(f"\nTest set class distribution:")
_, _, test_subtypes = load_dataset2_split('test')
from collections import Counter
class_counts = Counter(test_subtypes)
from configs.config import DS2_CLASSES

for idx, cls in enumerate(DS2_CLASSES):
    count = class_counts[idx]
    print(f"  {cls}: {count}")
