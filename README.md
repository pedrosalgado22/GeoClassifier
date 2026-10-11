# Geographical Classifier

This project attempts to create a Machine Learning model that predicts where a photo was taken, at different levels of measurement: country accuracy, continent accuracy, and kilometer distance by predicting the latitude/longitude.

<p align="center">
  <img
    width="456"
    height="421"
    alt="image-71_png"
    src="https://github.com/user-attachments/assets/ad5c9347-c019-4670-be5b-f3e72bb36c81"
  />
</p>

---



## Methodology

Instead of using NNs to learn and predict the target based on the images directly, as common image prediction models do, this project uses a much less efficient, accurate, and useful methodology: extracting tabular data from the images using different functions to derive values for different image characteristics, such as HSV, green color (VARI), sky proportion, HOG, LBP, ground color, brightness/contrast, and edge density, and then making the predictions. This method was chosen due to: first, NNs are black boxes that make it harder to understand everything that is going on, which is worse for learning. Second, I wanted to experiment with how efficient a model could be using tabular data to make geographical predictions, and how much worse it would be compared to the multiple existing ML/NN models that already make them at a very high level.

Classical algorithms expect `X` to be a 2D numeric table, not a raw image grid, so the entire first phase of this project was about deriving a meaningful, fixed-length numeric feature vector per image (hand-engineered features) rather than feeding in raw pixels. This approach mirrors and is inspired by the pre-deep-learning approach to this exact problem: the [IM2GPS project](https://graphics.cs.cmu.edu/projects/im2gps/) (Hays & Efros, 2008) tackled photo geolocation using engineered descriptors, such as color histograms, textures, and scene layouts, before CNN-based approaches became standard.

This methodology also has its many limitations: the obvious one is that, although transforming images into numeric matrices might be a fun idea, I have no expertise in image processing, whatever characteristics can be extracted from images or constitute them. That limitation served as a learning phase for understanding what constitutes an image, which is always a valuable skill for a data worker, particularly when evaluating photographic datasets.



# Repository

```
GeoClassifier/
├── imageconverter.py       # imageconverter.ipynb as a single script
├── imageconverter.ipynb    # Notebook that runs feature extraction across the dataset, outputs train/test CSVs
├── transformers.py         # Feature extraction functions imported by imageconverter.ipynb
├── clasifier.ipynb         # Cleaning, feature selection, scaling, modeling, tuning, ensembling, evaluation
├── test_set.csv            # Extracted/processed test features
├── train_set.csv           # Extracted/processed train features (gitignored)
├── README.md
├── LICENSE
├── .gitignore
└── __pycache__/       
```


## Dataset

**Source:** [sylshaw/streetview-by-country](https://www.kaggle.com/datasets/sylshaw/streetview-by-country/data) (Kaggle)

- ~107,000 images, 640×640 px, across more than a hundred countries/regions, with per-country file counts being near-uniform around 1,000 images, from the largest country in the world, Russia, to small territories such as Jersey or Christmas Island. Almost all territories present have 800–1,400 images regardless of real-life characteristics, with some countries having fewer (but all having at least 100+).
- Organized as `streetview_images/<ISO_ALPHA-2_CODE>/<filename>.jpg`.
- Filenames encode the exact capture coordinates and camera heading, e.g. `42d436079_1d473612_h213.jpg` is latitude `42.436079`, longitude `1.473612`, heading `213°`. An `n` prefix denotes a negative latitude/longitude (e.g. `n34d792510` = `-34.792510`); `d` stands in for the decimal point. Heading can also be negative (e.g. `h-75`).

## Data Pipeline
### 1. Data acquisition
Downloaded with `kagglehub` into a gitignored, repo-local `data/` folder (`output_dir`), so paths stay stable.

### 2. Metadata extraction
Each image's folder (ISO country code) and filename (coordinates, heading) are parsed with simple string splitting (`file_path.stem.split('_')`) into a dataframe: `country_code`, `image_name`, `width`, `height`, `size`, `path`, `latitude`, `longitude`, `heading`.

### 3. Label derivation and verification
`country` and `continent` are derived from each image's coordinates using `reverse_geocoder`, `pycountry` and `pycountry_convert`. The folder code is kept as `coord_country_code` for comparison. **3,572 images disagreed**, expected near borders and coastlines.

### 4. Class balancing
Per-country counts were inspected. Countries with fewer than 110 images were dropped.

### 5. Data quality checks
Tabular cleaning steps were adapted to images:

| Tabular check |
|---|
| Corrupted/missing data |
| Missing values |
| Mislabeled data |
| Duplicates |
| Inconsistent format 
| *(image-specific)* Placeholder frames |

Implemented as three sequential functions (`check_corrupted`, `check_channels`, `check_duplicates`), each taking a dataframe and returning it with a new column, with 0 corrupted images, 100% 3-channel, 0 duplicate hashes.


### Two real bugs found and fixed during this phase

1. **Namibia's ISO code is the literal string `"NA"`,** with every Namibian (and some neighboring-country border-adjacent) `country_code`/`coord_country_code` entry silently becoming an actual missing value — `"NA"`.
2. **OpenCV's Hue channel only spans 0–179, not 0–255.**

## Feature Engineering

| Feature | Count | What it measures in an image | Why it matters for country prediction |
|---|---|---|---|
| **HSV histogram** | 48 | Share of pixels per hue, saturation and brightness level (16 bins each) | Colour palette of the scene (soil, vegetation, buildings, sky). HSV separates colour from lighting.
| **Vegetation (VARI)** | 3 | How "green" the image is: mean, std, share of green pixels | Separates lush from arid or built-up regions|
| **Sky** | 2 | `sky_fraction`: how much of the top third is sky. `sky_blue_ratio`: how much of that sky is blue rather than grey | Openness vs. canopy or dense city, and sky type.|
| **HOG (PCA)** | 30 | Direction of edges in a 4×4 grid over the image (144 raw, reduced) | Dense urban grids vs. open rural roads, building layout |
| **LBP (PCA)** | 7 | Local texture patterns in a 2×2 grid (40 raw, reduced) | Surface texture: foliage vs. asphalt vs. dirt, and where each sits in the frame |
| **Ground colour** | 4 | Hue, saturation, brightness and grey share of the bottom third | Road surface material (grey asphalt, red laterite, dirt) |
| **Brightness / contrast** | 2 | Mean and spread of grayscale intensity | Atmosphere: hazy/overcast vs. clear/harsh light |
| **Edge density** | 2 | Amount of edges in the top and bottom half (Canny) | Busy vs. empty scenes |


- Sky and ground use fixed top/bottom thirds, a position proxy that fails under tree canopy or in urban canyons.
- HOG and LBP are resized to 128×128 for speed and could be reevaluated for better results with more computation. PCA is fit on train only  (80% variance retained).

## Feature Selection

**Redundancy (Pearson, threshold 0.8, feature vs. feature, evaluated manually beforehand):** adjacent HSV bins, LBP bins, `vari_mean`/`vari_std`, and `size`/`edge_density_total`/`edge_density_top`.

**Relevance to `country` (categorical target):** Pearson and Kendall don't apply, chi-square was skipped because it needs binned inputs. Two tests used:
- **ANOVA F-test:**
- **Mutual information:**

A feature was dropped if it fell in the **bottom 20% of both** rankings.



One stratified train/validation split instead of k-fold. Scaling and PCA were fit once on the full training split, not per fold.

This can cause minor data leakage because validation data indirectly influences the transformations but in a less rigorous experiment this is ok, provided the final test set remains completely separate and unseen. 

## Final Split

| Split | Rows |
|---|---|
| Train | 86,471 |
| Validation | 9,608 |
| Test | 10,676 |

80 features after cleaning, selection and transformation.

## Baseline Models (validation set)

Top-3 accuracy was tracked alongside top-1 ( where the correct country is in the top 3 prediction proba, as guessing Spain/Italy/Portugal and South Korea/Thailand/Cambodia for Portugal is NOT the same ).

| Model | Top-1 | Top-3 |
|---|---|---|
| Logistic Regression | 0.318 | 0.498 |
| Random Forest  | 0.296 | 0.456 |
| Naive Bayes | 0.187 | 0.337 |
| Decision Tree | 0.150 | 0.164 |
| KNN (k=5) | not recorded | not recorded |

- **KNN:** expected to do poorly. Distances lose meaning in ~80 mixed-scale dimensions, and KNN can't weight useful features.
- **Naive Bayes:** expected to do poorly. It assumes independent features, which the correlation analysis already contradicted.

- I expect XGBoost to beat these results but I wanted to maintain the simplicity theme that inspired this project and use the most basic models.

Both RF and LR were then tuned for maximum performance.

## Test Set

| Model | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| Random Forest | 0.337 | 0.441 | 0.505 |
| Logistic Regression | 0.315 | 0.428 | 0.495 |

## Ensembling

RF and LR probabilities are on different confidence scales. Two fixes compared:

- **Calibrated average:** isotonic calibration of each model on the validation set (`CalibratedClassifierCV` + `FrozenEstimator`), then averaged.
- **Stacking:** a small meta-LR trained on both models' probabilities.

| Ensemble | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| **Calibrated average** | **0.346** | **0.454** | **0.515** |
| Stacked | 0.334 | 0.439 | 0.506 |


### Baselines for comparison

A majority-class baseline (guessing the single most frequent country) for judging how much prediction power the ~31–35% top-1 / ~50–52% top-3 models have given the dataset's official counts is:

- **Top-1: ≈ 0.9–1.3%** (1,005–1,408 ÷ ~107,269)
- **Top-3: ≈ 2.8–4%**


Against the random baseline, the calibrated ensemble's 34.6% top-1 / 51.5% top-3 represents roughly a **30–40× improvement over random/majority-class guessing** which is strong evidence that the hand-engineered features carry real geographic signal.

## Qualitative Observations

Comparing the per-country accuracy table with the images shows that **countries whose images look alike internally score higher.**

- **São Tomé:** images share a blur throughout the island.
- **Oman:** a white car appears in a subset of images with similar landscapes and skies traditional of the non-urban middle east.
- **Tunisia:** similar scenes (clear sky, similar road type) with little cover usually only in main highways which promotes similarity.

This hypothesis also seems to confirm for the worst performing countries like Russia or the USA, massive countries with innumerous different biomes inside them and a very high coverage rate with a lot of different cars for different seasons and years, making pattern recognition for such a basic model extremely hard. Vietnam was the worst country however, which is curious since both the dataset photos and the general Vietnam appearance have less variance than some of the countries here.


## Extra Targets: Continent and Lat/Long

Both reuse the country pipeline, features and splits, as side experiments. The project was not planned for the targets at hand so both code and plan are less optimized for it, particularly the lat/long prediction as it uses regression, not classification output.

### Continent

Top-3 is meaningless here (random guessing already gets ~50%). **Majority baseline:** always guessing "Europe" scores around 40%.

| Model | Val Top-1 | Test Top-1 |
|---|---|---|
| Random Forest | 0.598 | 0.616 (tuned) |
| Logistic Regression | 0.528 | 0.435 (tuned) |
| Decision Tree | 0.457 | n/a |
| Naive Bayes | 0.351 | n/a |
| Calibrated average (RF + LR) | n/a | 0.611 |
| **Stacked (meta-LR)** | n/a | **0.627** |


- **Top features** (`lbp_pca_2`, `ground_hue_mean`, `ground_grey_frac`), matching the country ranking.
- **Per-continent top-1:** Africa 0.65, Europe 0.47, Oceania 0.29, South America 0.27, Asia 0.15, North America 0.14. South America jumps from 0.27 to 0.66 at top-2, showing its similarity to Asia in landscapes and nature settings.


### Lat/Long


| Model | Mean km | Median km |
|---|---|---|
| Linear Regression | 5,507 | 4,121 |
| Ridge | 5,507 | 4,121 |
| Decision Tree | 6,166 | 5,854 |
| Random Forest | 5,119 | 3,801 |
| Random Forest (tuned, test) | 5,132 | 3,773 |
| Ridge (tuned, test) | 5,528 | 4,126 |
| Simple average (RF + Ridge) | 5,303 | 3,895 |
| **Stacked (meta-model)** | **4,828** | **3,613** |


- **Mean runs 1,300–2,500 km above median** for every model.
- Even at eye test and without a clear baseline is pretty obvious this version has the lowest prediction power of the 3 (which is to be expected since this is the hardest task and at the same time the one the current notebook is least optimized for).

## Full-Project Summary and Comparison to Published Benchmarks

### This project's results, all three target levels

| Target | Metric | Best result | Model |
|---|---|---|---|
| Country (111 classes) | Top-1 / Top-2 / Top-3 accuracy | 34.6% / 45.4% / 51.5% | Calibrated ensemble (RF + LR) |
| Continent (6 classes) | Top-1 / Top-2 / Top-3 accuracy | 62.7% / 82.0% / 90.5% | Stacked ensemble (RF + LR) |
| Lat/Long | Mean / Median km error | 4,827.7 / 3,612.6 km | Stacked ensemble (RF + Ridge) |

### External benchmarks

| System | Year | Approach | 1 km | 25 km | 200 km | 750 km | 2,500 km |
|---|---|---|---|---|---|---|---|
| **IM2GPS** (Hays & Efros) | 2008 | Nearest-neighbor retrieval over hand-crafted features — **no neural network** | — | 12.0% | 15.0% | 23.0% | 47.0% |
| **IM2GPS, large-scale revisit** (Hays & Efros) | 2015 | Improved lazy-learning retrieval, still hand-crafted features — **no neural network** | 2.5% | 21.9% | 32.1% | 35.4% | 51.9% |
| **PlaNet** (Weyand, Kostrikov, Philbin) | 2016 | First CNN-based approach; classification over ~26,000 geographic cells | 8.4% | 24.5% | 37.6% | 53.6% | 71.3% |
| **ISNs** (Müller-Budack et al.) | 2018 | Deep CNN with hierarchical scene-context classification | 10.5% | 28.0% | 36.6% | 49.7% | 66.0% |
| **TransLocator** (Pramanick et al.) | 2022 | Transformer-based, combines image + generated semantic segmentation | 11.8% | 31.1% | 46.7% | 58.9% | 80.1% |


- **PIGEON** (Haas et al., CVPR 2024): a CLIP-based model tested on 5,000 GeoGuessr-style Street View images, the closest data to this project's. Median error **44 km**, **~92%** correct country, **~40%** of guesses within 25 km. It beat a professional GeoGuessr player in six straight matches. *(arxiv.org/abs/2307.05845)*
- **PlaNet** (Weyand et al., 2016): won 28 of 50 rounds against 10 expert GeoGuessr players, with a median error of **1,100–2,300 km** in that format. *(via Anderson et al., par.nsf.gov/servlets/purl/10053400)*

### Comparing this project's numbers to the above

**Caveats**

1. **"% within 750 km" is a mean proxy for "correct country,"**, while this project is a literal label match, which is stricter, but means that comparisons with IM2GPS are approximations. PIGEON's 92% is the one directly comparable figure, since it is also a literal country match.
2. **Training scale differs by orders of magnitude.** PlaNet trained on 91M images, and PIGEON's CLIP backbone was pretrained on hundreds of millions of image-text pairs. This project used ~86k images, classical models on hand-engineered features, and a consumer CPU.

**Results**
- **Country:** 34.6% sits between the 2008 IM2GPS (23.0% on the 750 km proxy) and its 2015 revision (35.4%), on a stricter metric. It is well below PlaNet (53.6%) and TransLocator (58.9%), and far below PIGEON (92%).
- **Continent:** 62.7% is close to ISNs (66.0%, a 2018 deep-learning system) and above both IM2GPS variants (47.0%, 51.9%). It is below PlaNet (71.3%) and TransLocator (80.1%). This is the project's strongest showing against deep-learning benchmarks.
- **Distance:** the weakest showing. The best median error (3,613 km) is **1.6-3.3x worse** than PlaNet's 1,100-2,300 km (a 2016 system) and **~82x worse** than PIGEON's 44 km. Every model's typical miss exceeds 2,500 km, the continent threshold that the papers above use as their most forgiving bucket.

**Final thought:** The final model is in my opinion worse than these results might suggest. The Uniform distribution in the data made the model's job on accuracy a lot easier, since it gave small countries and territories the same weight as it did for giant countries, which allowed the model to get really good at recognizing natural and image-adherent patterns present in small countries and/or countries with very small Google Maps coverage and obvious camera quirks. In a truly representative dataset of the world's landmass, the model would do much much worse.


## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| `pandas` / `numpy` | Data handling |
| `Pillow` (`PIL`) / `opencv-python` (`cv2`) | Image loading and processing |
| `scikit-image` (`skimage.feature`) | HOG and Local Binary Pattern extraction |
| `scikit-learn` | Models (LR, DT, RF, KNN, NB), PCA, scaling, feature selection (`f_classif`, `mutual_info_classif`), `GridSearchCV`/`RandomizedSearchCV`, `CalibratedClassifierCV`/`FrozenEstimator`, metrics (`accuracy_score`, `top_k_accuracy_score`), `learning_curve`, `permutation_importance` |
| `reverse_geocoder` | Coordinate → country code |
| `pycountry` / `pycountry-convert` | Country code → name / continent |
| `imagehash` | Perceptual-hash duplicate detection |
| `matplotlib` / `plotly` | Visualizations |
| `kagglehub` | Dataset acquisition |
| `joblib` | Disk-caching of expensive extraction cells across notebook restarts |
| Jupyter / VS Code | Development environment |
| Git / GitHub | Version control |
| Claude (Anthropic) | AI
## Acknowledgments

- Dataset: [sylshaw/streetview-by-country](https://www.kaggle.com/datasets/sylshaw/streetview-by-country/data)
- Structural inspiration for the metadata-extraction pipeline: [crypticsy — "Geo Guesser"](https://www.kaggle.com/code/crypticsy/geo-guesser) by Animesh Singh Basnet
- Learning progression and pedagogical approach: Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — [ageron/handson-ml3](https://github.com/ageron/handson-ml3)
- Classical feature-engineering approach grounded in Hays & Efros, *IM2GPS: Estimating Geographic Information from a Single Image* (2008)

## License / Usage Note

The source dataset's license is listed as "Unknown" on Kaggle, and Google's Maps Platform terms restrict model training on Street View content more broadly. This project is personal and educational; it is not intended for commercial use or redistribution of the underlying imagery.
