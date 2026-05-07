import os
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Shuffle multi-modal embeddings while keeping the image-text pair intact.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing image_feat.npy and text_feat.npy")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for deterministic shuffling")
    args = parser.parse_args()

    np.random.seed(args.seed)

    img_path = os.path.join(args.data_dir, 'image_feat.npy')
    txt_path = os.path.join(args.data_dir, 'text_feat.npy')

    features = {}
    if os.path.exists(img_path):
        features['img'] = np.load(img_path)
    if os.path.exists(txt_path):
        features['txt'] = np.load(txt_path)

    if not features:
        print("No features found to shuffle.")
        return

    num_items = None
    for k, v in features.items():
        if num_items is None:
            num_items = len(v)
        else:
            assert len(v) == num_items, "Mismatch in number of items between modalities!"

    first_feat = list(features.values())[0]
    is_pad = bool(np.all(first_feat[0] == 0))

    if is_pad:
        perm = np.arange(num_items)
        perm[1:] = np.random.permutation(np.arange(1, num_items))
    else:
        perm = np.random.permutation(num_items)

    print(f"Shuffling {num_items} items (kept index 0 intact: {is_pad}) in {args.data_dir}...")

    if 'img' in features:
        np.save(img_path, features['img'][perm])
        print("- Saved shuffled image_feat.npy")
    
    if 'txt' in features:
        np.save(txt_path, features['txt'][perm])
        print("- Saved shuffled text_feat.npy")

if __name__ == '__main__':
    main()
