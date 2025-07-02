import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random
import pandas as pd
import textwrap
import pdb
import numpy as np
import argparse
from open_clip import IMAGENET_CLASSNAMES

# top-level
def plot_images(images_dir,
                captions=None,
                include_predictions=True,
                randomize=True,
                num_to_plot=10,
                ids=None,
                top_k=1,
                pmi_metadata=None):

    predictions = Path(images_dir) / "predictions_ViT-B-32-quickgelu.csv"
    predictions_df = pd.read_csv(predictions)
    predictions_df = predictions_df[(predictions_df['target'] == predictions_df['pred1'])]
    if isinstance(predictions_df['image_id'].iloc[0], str):
        predictions_df['image_id'] = [int(x[7:-1]) for x in predictions_df['image_id']] # image ids saved as 'tensor(xxxx)'
    print("HELLO there!", flush=True)
    img_paths = [Path(images_dir) / str(predictions_df.loc[predictions_df['image_id'] == _id, 'target'].values[0]).zfill(3) / (str(_id) + ".png") for _id in predictions_df['image_id']]
    
    if ids is not None:
        img_paths = [Path(images_dir) / str(predictions_df.loc[predictions_df['image_id'] == _id, 'target'].values[0]).zfill(3) / (str(_id) + ".png") for _id in ids]
    elif randomize:
        img_paths = random.sample(img_paths, num_to_plot)
    else:
        img_paths = img_paths[:num_to_plot]

    if captions is not None:
        plot_images_with_caption(img_paths, captions, predictions={'clip': predictions} if include_predictions else None, top_k=top_k, pmi_metadata=pmi_metadata)
    else:
        fig, axs = plt.subplots(2,5, figsize=(12, 6))
        for i, img_path in enumerate(img_paths):
            imgnet_id = int(img_path.stem)
            if predictions is not None:
                predictions = pd.read_csv(predictions)
                img_preds = predictions.loc[predictions['image_id'] == imgnet_id]
            imgnet_class = int(img_path.parent.stem)
            ax = axs[i // 5, i % 5]
            ax.imshow(Image.open(img_path))
            ax.axis('off')
            ax.set_title(f"{IMAGENET_CLASSNAMES[imgnet_class]}/{IMAGENET_CLASSNAMES[img_preds['pred1'].values[0]]}")

def plot_images_with_caption(image_paths, captions, predictions={}, top_k=1, pmi_metadata=None):
    captions = pd.read_csv(captions, comment='#')
    if len(predictions) > 0:
        predictions = {k: pd.read_csv(v) for k, v in predictions.items()}
    fig = plt.figure(figsize=(10, 20))
    gs = fig.add_gridspec(10, 2, width_ratios=[2, 1])
    plotted_captions = []
    
    for i, img_path in enumerate(image_paths):
        extra_space = 0.02*i
        imgnet_id = int(img_path.stem)
        img_metadata = captions.loc[captions['word_pair_id'] == imgnet_id]
        img_caption = img_metadata.caption.values[0]
        img_caption = '\n'.join(textwrap.wrap(img_caption, width=60, break_long_words=False))
        imgnet_class = int(img_metadata.assigned_label_int.values[0])
        
        ax_text = fig.add_subplot(gs[i, 0])
        ax_text.text(0, 0.5+extra_space, img_caption, fontsize=11) #verticalalignment='center')
        result_str = f"Pair Word: {img_metadata.word_pair.values[0].split(',')[0]}, Target: {IMAGENET_CLASSNAMES[imgnet_class]}\nPredictions:"
        if pmi_metadata is not None:
            result_str += f"\nPMI: {pmi_metadata.loc[pmi_metadata['word_pair_id'] == imgnet_id, 'pmi_smooth_uni'].values[0]:.2f}"
        ax_text.text(0, 0.2+extra_space, result_str, fontsize=11)

        if len(predictions) > 0:
            for name, pred_df in predictions.items():
                if isinstance(pred_df['image_id'].iloc[0], str):
                    pred_df['image_id'] = [int(x[7:-1]) for x in pred_df['image_id']] # image ids saved as 'tensor(xxxx)'
                img_preds = pred_df.loc[pred_df['image_id'] == imgnet_id]
                starting_y = 0.2+extra_space
                for k in range(top_k):
                    pred_class = int(img_preds[f'pred{k+1}'].values[0])
                    pred_value = img_preds[f'prob{k+1}'].values[0]
                    ax_text.text(0.22, starting_y - 0.05*k, f"{IMAGENET_CLASSNAMES[pred_class]} ({pred_value:.2f})", color='green' if pred_class == imgnet_class else 'red', fontsize=11)
        ax_text.axis('off') 

        ax_image = fig.add_subplot(gs[i, 1])
        ax_image.imshow(Image.open(img_path))
        ax_image.axis('off')
        plotted_captions.append(img_caption)

    print('\n'.join(plotted_captions))
    plt.show()


def plot_images_and_prediction(image_dir, metadata):
    data_dir = Path(image_dir)
    data = metadata.sample(10)

    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    for i, pair in enumerate(data.itertuples()):
        path = data_dir / str(pair.target).zfill(3) / (str(pair.image_id) + ".png")
        predicted = IMAGENET_CLASSNAMES[pair.pred1]

        ax = axs[i // 5, i % 5]
        ax.imshow(Image.open(path), aspect='auto')
        ax.axis('off')
        ax.set_title(f"{pair.word_pair_final} \n {predicted}")

def plot_image(image_id, image_class, image_dir):
    data_dir = Path(image_dir)
    path = data_dir / str(image_class).zfill(3) / (str(image_id) + ".png")
    return Image.open(path)

def class_level_acc(df, zero_freq):
    high_freq_threshold = 1000

    # calculate accuracy with only high(low) freq pairs via 'include' column
    freqs = pd.read_csv("data/pair_to_imagenet_label/nonzero_freq_pair_to_imagenet_label__llm_filtered.csv")
    freqs['other_word'] = [x.split(",")[0] for x in freqs['word_pair_final']]
    df['object'] = [x.split("_")[0] for x in df['image_id']]
    
    unique_pairs = df[['object', 'target']].drop_duplicates()
    if zero_freq:
        # Get unique object-target pairs and their inclusion status
        unique_pairs['include'] = unique_pairs.apply(
            lambda row: row['object'] not in freqs[freqs['assigned_label_int'] == row['target']]['other_word'].tolist(),
            axis=1
        )
    else:
        freqs = freqs[freqs['frequency'] > high_freq_threshold]
        unique_pairs['include'] = unique_pairs.apply(
            lambda row: row['object'] in freqs[freqs['assigned_label_int'] == row['target']]['other_word'].tolist(),
            axis=1
        )
    df = df.merge(unique_pairs, on=['object', 'target'], how='left')
    
    correct = 0
    df = df[df['include']]
    for target in df['target'].unique():
        rows = df[df['target'] == target]
        if len(rows[rows['pred1'] == rows['target']]) == len(rows):
            correct += 1
    return correct / len(df['target'].unique())

def pmi(pairs_df):
    # get single word freqs
    all_nonzero_pairs = pd.read_csv("data/laion400m-meta/cleaned/pairs.csv") # all pairs with nonzero frequency in LAION-400M
    all_nonzero_pairs['word1'] = [x.split()[0] for x in all_nonzero_pairs['word_pair']]
    all_nonzero_pairs['word2'] = [x.split()[1] for x in all_nonzero_pairs['word_pair']]
    all_nonzero_pairs = all_nonzero_pairs[all_nonzero_pairs['word1'] != all_nonzero_pairs['word2']]

    melted = pd.concat([
        all_nonzero_pairs[['word1', 'frequency']].rename(columns={'word1': 'word'}),
        all_nonzero_pairs[['word2', 'frequency']].rename(columns={'word2': 'word'}),
    ], ignore_index=True)
    melted = melted[melted['word'].isin(pd.unique(all_nonzero_pairs['word1'].tolist() + all_nonzero_pairs['word2'].tolist()))]
    word_to_freq = melted.groupby('word', sort=False).sum().reset_index()

    if 'words_in_pairs' not in pairs_df.columns:
        pairs_df['words_in_pairs'] = pairs_df['pairs'].apply(lambda x: [y.strip("'").split(",") for y in eval("[" + x.strip("[]").replace(" ", "") + "]") if isinstance(y, str)])
    else:
        pairs_df['words_in_pairs'] = pairs_df['words_in_pairs'].apply(lambda x: eval(x))
    print(pairs_df.head()['words_in_pairs'])

    # Create word frequency lookup dictionary for faster access
    word_freq_dict = dict(zip(word_to_freq['word'], word_to_freq['frequency']))

    pairs_df['single_word_freqs_new'] = pairs_df['words_in_pairs'].progress_apply(
        lambda pair_list: [[word_freq_dict.get(word.lower(), 0) for word in pair] for pair in pair_list]
    )

    # calculate laplace smoothed PMI
    unigram_smoothing_factor = 1e3
    pair_smoothing_factor = 1

    pairs_df['uni_smoothed_freqs'] = pairs_df['single_word_freqs_new'].apply(lambda freqs_list_list: [[int(freq)+unigram_smoothing_factor for freq in freqs_list] for freqs_list in eval(freqs_list_list)])
    pairs_df['pmi_denominator_smoothed'] = pairs_df['uni_smoothed_freqs'].apply(lambda x: [np.prod(y) for y in x])

    total_unigrams = 3.2e9
    total_pairs = total_unigrams * (total_unigrams - 1) / 2
    total_pairs_smoothed = total_pairs + (total_unigrams * (total_unigrams - 1)) / 2
    const = (1/total_pairs_smoothed) / ((1/total_unigrams)**2)

    pairs_df['pmi'] = [[np.log(const * (joint+pair_smoothing_factor) / prod) for joint, prod in zip(eval(joints), prods)] for joints, prods in zip(pairs_df['laion_freqs'], pairs_df['pmi_denominator_smoothed'])]
    pairs_df['avg_pmi'] = pairs_df['pmi'].apply(lambda x: sum(x)/len(x))
    
    return pairs_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", type=str, required=True)
    args = parser.parse_args()

    pairs_df = pd.read_csv(args.pairs_csv)
    pmi(pairs_df)
