# %%
import os
import pathlib
from pathlib import Path
from math import ceil, sqrt

import numpy as np
import pandas as pd
import cv2
import imagehash
from PIL import Image

import reverse_geocoder as rg
import pycountry
import pycountry_convert as pc

import kagglehub

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

# %%
DATA_PATH = Path("data/images/streetview_images")

if not any(DATA_PATH.iterdir()):
    path = kagglehub.dataset_download(
        "sylshaw/streetview-by-country",
        output_dir=str(DATA_PATH)
    )
else:
    path = str(DATA_PATH)

print("Path to dataset:", path)

# %%
geo_data_dir = pathlib.Path(path)


def extract_metadata_from_folder(path):
    metadata = []
    for file_path in path.glob('*'):   # loops through every item inside path
        width, height = Image.open(file_path).size

        lat_str, lon_str, heading_str = file_path.stem.split('_')
        latitude = float(lat_str.lstrip('n').replace('d', '.'))
        if lat_str.startswith('n'):
            latitude = -latitude
        longitude = float(lon_str.lstrip('n').replace('d', '.'))
        if lon_str.startswith('n'):
            longitude = -longitude
        heading = int(heading_str[1:])
        if heading_str.startswith('n'):
            heading = -heading

        metadata.append({
            'country_code': path.name,
            'image_name': file_path.name,
            'width': width,
            'height': height,
            'size': file_path.stat().st_size,
            'path': file_path,
            'latitude': latitude,
            'longitude': longitude,
            'heading': heading
        })
    return metadata



# %%
# List all country folders
country_folders = [f for f in os.listdir(geo_data_dir) if os.path.isdir(os.path.join(geo_data_dir, f))]

# Initialize an empty list to store metadata
all_metadata = []

# Iterate through country folders and extract metadata
for country_name in country_folders:
    metadata = extract_metadata_from_folder(pathlib.Path(os.path.join(geo_data_dir, country_name)))
    all_metadata.extend(metadata)

# Create a Pandas DataFrame from the metadata
df_geo_data = pd.DataFrame(all_metadata)

def get_country_codes(df):
    coords = list(zip(df['latitude'], df['longitude']))
    return [r['cc'] for r in rg.search(coords, mode=1)]

def get_country(codes):
    return [pycountry.countries.get(alpha_2=c).name if pycountry.countries.get(alpha_2=c) else c for c in codes]

def get_continent(codes):
    result = []
    for c in codes:
        try:
            result.append(pc.convert_continent_code_to_continent_name(pc.country_alpha2_to_continent_code(c)))
        except KeyError:
            result.append(None)
    return result

coord_codes = get_country_codes(df_geo_data)
df_geo_data['country'] = get_country(coord_codes)
df_geo_data['continent'] = get_continent(coord_codes)

df_data_distribution = df_geo_data.groupby('country')['image_name'].count().reset_index().rename(columns={'image_name': 'frequency'})

df_geo_data.head()

# %%
df_geo_data['coord_country_code'] = coord_codes

mismatches = df_geo_data[df_geo_data['coord_country_code'] != df_geo_data['country_code']]

print(f"{len(mismatches)} mismatches")

mismatches[['image_name', 'country_code', 'coord_country_code', 'country']]

print(df_geo_data.isna().sum())

# %%
df_geo_data[df_geo_data['continent'].isna()]

# %% [markdown]
# These 2 will be dropped so is ok

# %%
# select a random country
random_country = df_geo_data.sample(1)

with Image.open(random_country.path.values[0]) as img:
    fig = go.Figure()
    fig.add_trace(go.Image(z=img))
    fig.update_layout(
        title="Image from the training dataset",
        template="plotly_white",
        width=900,
        height=450
    )
    
    fig.update_xaxes(title_text="Width")
    fig.update_yaxes(title_text="Height")
    
    fig.add_annotation(
        x=0.5,
        y=1.1,
        xref="paper",
        yref="paper",
        text="Country : {}".format(random_country.country.values[0]),
        showarrow=False
    )
    
    fig.show()

# %%
print("The total number of images in the dataset are {:,} images.".format(df_data_distribution.frequency.sum()))
print("The average number of images per country is {:,} images.".format(int(df_data_distribution.frequency.mean())))
print("The median number of images per country is {:,} images.".format(int(df_data_distribution.frequency.median())))
print("25% of the countries have {:,} images or less, while 75% of the countries have {:,} images or more.".format(int(df_data_distribution.frequency.quantile(0.25)), int(df_data_distribution.frequency.quantile(0.75))))
print("The maximum and minimum number of images per country are {:,} and {:,} images respectively.".format(df_data_distribution.frequency.max(), df_data_distribution.frequency.min()))

# %%
# Function to plot heatmap
def plot_heatmap(df, title_text, color_threshold=0.90):
    # Shaping Data to fit graph by finding the max size of the grid to fit all the images
    grid_size = ceil(sqrt(df.shape[0]))**2
    
    pad = '-'
    n_labels = df.country.to_list() + ([pad] * (grid_size - df.shape[0]))
    n_samples = df.frequency.to_list() + ([np.nan] * (grid_size - df.shape[0]))
    
    samples = np.split(np.array(n_samples), sqrt(grid_size))
    
    fig = go.Figure(data=go.Heatmap(
        z=samples,
        x0=0,
        dx=1,
        y0=0,
        dy=1,
        colorscale='Emrld',
        text=n_labels,
        hoverinfo='text',
    ))
    
    # Add labels within the heatmap
    for i, country in enumerate(df['country']):
        x_pos = i % int(sqrt(grid_size))
        y_pos = i // int(sqrt(grid_size))
        fig.add_annotation(
            x=x_pos,  # Center the label within the grid cell
            y=y_pos,
            text= "{}<br> Count: {}".format(n_labels[i].replace(' ', '<br>'), n_samples[i]),
            showarrow=False,
            font=dict(
                color='#474747' if n_samples[i] < df.frequency.quantile(color_threshold) else 'white',
                size=10),
        )
    
    # hide axis labels
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange='reversed')
    
    # change the dimensions of the figure
    fig.update_layout(
        title=title_text,
        height=1200,
        width=1200
    )
    fig = add_plot_styling(fig)
    
    # Use iplot to display the graph in the Jupyter Notebook
    pyo.iplot(fig)

# %%
# Function to add custom styling to the plot
def add_plot_styling(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0.05)',  # Set background color to transparent
    )
    return fig



# %%
plot_heatmap(df_data_distribution.sort_values(by='frequency', ascending=True), "Images per Country", color_threshold=0.98)

# %%
df_geo_data.keys()

# %%
# outlier removal
filtered_countries = df_data_distribution[(df_data_distribution.frequency >= 110) & (df_data_distribution.frequency <= 10000) ].country.unique()

df_filtered_geo_data = df_geo_data[df_geo_data.country.isin(filtered_countries)]


# %%
plot_heatmap(df_filtered_geo_data.groupby('country')['image_name'].count().reset_index().rename(columns={'image_name': 'frequency'}).sort_values(by='frequency', ascending=True), "Images per Country", color_threshold=0.98)

# %%
def check_corrupted(df):
    results = []
    for path in df['path']:
        img = cv2.imread(str(path))
        results.append(img is None)
    df['corrupted'] = results
    return df

# %%
def check_channels(df):
    results = []
    for path in df['path']:
        img = cv2.imread(str(path))
        results.append(img.shape[2] if img.ndim == 3 else 1)
    df['channels'] = results
    return df

# %%
def check_duplicates(df):
    results = []
    for path in df['path']:
        results.append(str(imagehash.phash(Image.open(path))))
    df['phash'] = results
    return df

# %%
import joblib
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def cached(func, name, df, *args, **kwargs):
    cache_path = CACHE_DIR / f"{name}.pkl"
    if cache_path.exists():
        return joblib.load(cache_path)
    result = func(df, *args, **kwargs)
    joblib.dump(result, cache_path)
    return result

# %%
df_filtered_geo_data = cached(check_corrupted, "corrupted", df_filtered_geo_data)
df_filtered_geo_data = cached(check_channels, "channels", df_filtered_geo_data)
df_filtered_geo_data = cached(check_duplicates, "duplicates", df_filtered_geo_data)

print(df_filtered_geo_data['corrupted'].sum(), "corrupted images")
print(df_filtered_geo_data['channels'].value_counts())
print(df_filtered_geo_data['phash'].duplicated(keep=False).sum(), "images with a duplicate hash")

# %%
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_index, test_index = next(splitter.split(df_filtered_geo_data, df_filtered_geo_data["country"]))

train_set = df_filtered_geo_data.iloc[train_index]
test_set = df_filtered_geo_data.iloc[test_index]

len(train_set)

# %%
len(test_set)

# %%
from transformers import color_histogram, vegetation_features, extract_edges, sky_features, extract_hog, extract_lbp, extract_ground, extract_contrast

# %%
train_features = [color_histogram(path) for path in train_set['path']]
test_features = [color_histogram(path) for path in test_set['path']]

train_hsv_cols = pd.DataFrame(train_features, columns=[f'hsv_bin_{i}' for i in range(48)])
test_hsv_cols = pd.DataFrame(test_features, columns=[f'hsv_bin_{i}' for i in range(48)])

train_set = pd.concat([train_set.reset_index(drop=True), train_hsv_cols], axis=1)
test_set = pd.concat([test_set.reset_index(drop=True), test_hsv_cols], axis=1)

# %%
print(any(c.startswith('hsv_') for c in train_set.columns))

# %% [markdown]
# i doubt a green detector will have a lot of differentiating power since in google view all countries will have a lot of green and it is not super helpful if not allowing to distinnguish each but well see later with eature importance does not hurt
# 
# 

# %%
train_veg = [vegetation_features(path) for path in train_set['path']]
test_veg = [vegetation_features(path) for path in test_set['path']]

train_set = pd.concat([train_set.reset_index(drop=True), pd.DataFrame(train_veg, columns=['vari_mean','vari_std','vari_frac'])], axis=1)
test_set = pd.concat([test_set.reset_index(drop=True), pd.DataFrame(test_veg, columns=['vari_mean','vari_std','vari_frac'])], axis=1)

# %%
print(any(c.startswith('vari_') for c in train_set.columns))

# %% [markdown]
# sky proportion will mainly be a proxy for is this south africa or australia vs anywhere else really 
# 
# 
# "Upper third of frame" only approximates sky location because street-view cameras are roughly fixed-height, vehicle-mounted shots — horizon tends to sit near frame-center, so sky (when visible) concentrates in the top third.

# %%
train_sky = [sky_features(path) for path in train_set['path']]
test_sky = [sky_features(path) for path in test_set['path']]

train_set = pd.concat([train_set.reset_index(drop=True), pd.DataFrame(train_sky, columns=['sky_fraction','sky_blue_ratio'])], axis=1)
test_set = pd.concat([test_set.reset_index(drop=True), pd.DataFrame(test_sky, columns=['sky_fraction','sky_blue_ratio'])], axis=1)

# %%
print(any(c.startswith('sky_') for c in train_set.columns))

# %%
train_set = cached(extract_hog, "train_hog", train_set)
test_set = cached(extract_hog, "test_hog", test_set)

# %%
print(any(c.startswith('hog_') for c in train_set.columns))

# %%
from sklearn.decomposition import PCA
import numpy as np

def fit_full_pca(df):
    hog_cols = [c for c in df.columns if c.startswith('hog_')]
    pca = PCA()
    pca.fit(df[hog_cols])
    return pca

# %%
pca = fit_full_pca(train_set)

# %%
# Kaiser Criterion
eigenvalues = pca.explained_variance_
print("Eigenvalues:")
print(eigenvalues)
print(f"Components with eigenvalue > 1: {(eigenvalues > 1).sum()}")

# %%
# Cumulative Variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
print("Cumulative Variance:")
print(cumulative_variance)
print(f"Components for 80% variance: {np.argmax(cumulative_variance >= 0.80) + 1}")

# %%
# Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o', linestyle='-', color='blue')
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.xlim(0, 30)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
n_pca_components = 30  # set after reading Cell 3

def apply_pca_to_hog(df, n_components):
    hog_cols = [c for c in df.columns if c.startswith('hog_')]
    pca = PCA(n_components=n_components)
    hog_pca = pca.fit_transform(df[hog_cols])
    hog_pca_cols = pd.DataFrame(hog_pca, columns=[f'hog_pca_{i}' for i in range(hog_pca.shape[1])])
    df = pd.concat([df.drop(columns=hog_cols).reset_index(drop=True), hog_pca_cols], axis=1)
    return df, pca

def transform_pca_hog(df, pca):
    hog_cols = [c for c in df.columns if c.startswith('hog_')]
    hog_pca = pca.transform(df[hog_cols])
    hog_pca_cols = pd.DataFrame(hog_pca, columns=[f'hog_pca_{i}' for i in range(hog_pca.shape[1])])
    df = pd.concat([df.drop(columns=hog_cols).reset_index(drop=True), hog_pca_cols], axis=1)
    return df

# %%
train_set, pca = apply_pca_to_hog(train_set, n_pca_components)
test_set = transform_pca_hog(test_set, pca)

# %%
print(any(c.startswith('hog_pca_') for c in train_set.columns))

# %%
from skimage.feature import local_binary_pattern
from transformers import extract_lbp


# %%
train_set = cached(extract_lbp, "train_lbp", train_set)
test_set = cached(extract_lbp, "test_lbp", test_set)

# %%
print(any(c.startswith('lbp_') for c in train_set.columns))

# %%
train_set = extract_ground(train_set)
test_set = extract_ground(test_set)

# %%
print(any(c.startswith('ground_') for c in train_set.columns))

# %%
train_set = extract_contrast(train_set)
test_set = extract_contrast(test_set)

# %%
print(any(c.startswith('brightness_') or c.startswith('contrast_') for c in train_set.columns))

# %%
train_set = extract_edges(train_set)
test_set = extract_edges(test_set)

# %%
print(any(c.startswith('edge_') for c in train_set.columns))

# %%
train_set.to_csv("train_set.csv", index=False)
test_set.to_csv("test_set.csv", index=False)


