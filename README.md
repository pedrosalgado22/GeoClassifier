# GeoClassifier

A machine learning project that predicts a country from a single street-view-style photo — the core mechanic of [GeoGuessr](https://www.geoguessr.com/), rebuilt as a supervised image classification problem.

## Motivation

I play GeoGuessr. One of the goals of this project, beyond the learning exercise itself, is to eventually have a model that can go head-to-head with me — given one frame, guess the country, and see who's more consistent.

## Goal

Given a single street-view image, predict the country it was taken in (and, as a coarser fallback target, the continent). The project is being built incrementally, prioritizing understanding over shortcuts to a high-accuracy model.

## Learning Philosophy

This is my first project working with image data rather than tabular data, and my first time approaching neural networks in any depth. Rather than jumping straight to a CNN — which would work, but as a black box I don't yet have the intuition to reason about — the project deliberately follows the progression laid out in Aurélien Géron's *Hands-On Machine Learning* ([ageron/handson-ml3](https://github.com/ageron/handson-ml3)): build fluency with classical supervised learning (Random Forest, Logistic Regression, Naive Bayes) first, on data I already know how to structure as `X`/`y`, before moving into neural network architectures.

The catch with classical algorithms is that they expect `X` as a 2D numeric table, not a raw image grid — so the current phase of the project is about deriving a meaningful numeric feature vector per image (hand-engineered features), rather than feeding pixels in directly. This mirrors the pre-deep-learning approach to this exact problem: the [IM2GPS project](https://graphics.cs.cmu.edu/projects/im2gps/) (Hays & Efros, 2008) tackled photo geolocation using engineered descriptors — color histograms, texture, scene layout — before CNN-based approaches became standard. This project is, in effect, a scaled-down rebuild of that baseline, as a stepping stone toward CNNs once the fundamentals are solid.

## Dataset

**Source:** [sylshaw/streetview-by-country](https://www.kaggle.com/datasets/sylshaw/streetview-by-country/data) (Kaggle)

- ~107,000 images, 640×640 px, across 111 countries/regions
- Balanced by design — roughly 1,000 images per country, addressing the class-imbalance problem common in similar datasets
- Organized as `streetview_images/<ISO_ALPHA-2_CODE>/<filename>.jpg`
- Filenames encode the exact capture coordinates and camera heading, e.g. `42d436079_1d473612_h213.jpg` → latitude `42.436079`, longitude `1.473612`, heading `213°` (an `n` prefix denotes a negative value; `d` stands in for the decimal point)

### Dataset selection history

Two other datasets were evaluated and abandoned before settling on the above:

1. **[ubitquitin/geolocation-geoguessr-images-50k](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k)** — the dataset used in [crypticsy's "Geo Guesser" notebook](https://www.kaggle.com/code/crypticsy/geo-guesser) by Animesh Singh Basnet, which this project's metadata-extraction pattern is directly adapted from. Abandoned due to severe class imbalance (the US alone accounts for ~12,000 images, while dozens of countries have single-digit counts), which would bias any classifier toward a handful of over-represented countries.
2. **[paulchambaz/google-street-view](https://www.kaggle.com/datasets/paulchambaz/google-street-view)** (10k images) — abandoned because the accompanying coordinates file had no reliable, confirmed key linking rows to specific image files.

## Methodology (current state)

### 1. Data acquisition
Downloaded via `kagglehub`, targeted into a repo-local, gitignored `data/` directory rather than the default global cache, so the project path is stable and reproducible across machines.

### 2. Metadata extraction
Every image's folder (ISO country code) and filename (coordinates + heading) are parsed into a single dataframe: `country_code`, `image_name`, `width`, `height`, `size`, `path`, `latitude`, `longitude`, `heading`.

### 3. Label derivation and verification
Rather than trusting the folder-provided country code alone, `country` and `continent` labels are derived directly from each image's actual coordinates via `reverse_geocoder`, `pycountry`, and `pycountry_convert`. The folder-provided code is retained alongside the coordinate-derived one as a QA check — of ~107k images, ~3,572 disagree between the two, which is expected for points near borders or coastlines, and worth spot-checking rather than silently trusting either source blindly.

### 4. Exploratory data analysis and class balancing
Distribution of images per country was inspected (mean, median, quartiles, and a heatmap visualization), and countries with fewer than 110 images were filtered out to keep the class distribution usable for training.

### 5. Feature engineering — current decision point
Since classical algorithms need a fixed-length numeric vector per image rather than a raw pixel grid, the project is building a hand-engineered feature set, one feature family at a time. Candidates under consideration: color histograms, vegetation index, sky proportion, HOG (edge/shape structure), LBP (local texture), dominant line orientation, GIST, OCR-based text/script detection, sun/shadow angle, and color temperature.

**Implemented so far:** HSV color histogram, 16 bins per channel (48 features per image), extracted in parallel across all ~107k images via `multiprocessing.Pool`. HSV was chosen over RGB specifically to avoid encoding the same color information twice (the two are a deterministic transform of each other, so including both would be redundant rather than additive — and actively harmful for Logistic Regression, which is sensitive to correlated inputs) and because separating hue from brightness/saturation is more robust to the inconsistent lighting/exposure across street-view captures than raw RGB.

## Current status

This README reflects the project up to the HSV color histogram feature — the first entry in the engineered feature set. Additional features (HOG, LBP, and others from the candidate list above), baseline model training (Random Forest, Logistic Regression, Naive Bayes), and evaluation are not yet implemented, and CNN work is intentionally deferred until the classical pipeline is complete and understood.


**Implemented so far:**

- **HSV color histogram** (16 bins per channel, 48 features total). HSV was chosen over RGB to avoid encoding the same color information twice — the two are a deterministic transform of each other, so extracting both would be redundant rather than additive, and actively harmful for Logistic Regression, which is sensitive to correlated inputs. Separating hue from brightness/saturation also makes the feature more robust to the inconsistent lighting/exposure across street-view captures than raw RGB would be.

- **Vegetation index (VARI)**. True NDVI requires a near-infrared channel, which street-view RGB photos don't have, so VARI (`(G-R)/(G+R-B)`) is used as the standard visible-band proxy from the precision-agriculture literature. Using a ratio rather than raw channel values makes the signal resistant to illumination differences across images — the same design goal as the HSV choice above, solved the same way (relative/normalized values instead of raw magnitudes). Three summary statistics are taken from the resulting per-pixel score: mean (overall vegetation amount), standard deviation (how uniformly vegetated the frame is), and the fraction of pixels above a threshold (vegetation coverage extent). This feature is included with the expectation that it will only meaningfully separate climate extremes (desert nations, dense city-states, tropical nations) rather than most of the 111-class set — its actual contribution will be checked against Random Forest feature importances once a baseline model exists, rather than assumed.
**Sky proportion.** Restricted to the top third of the frame (a positional proxy, not true segmentation — street-view cameras sit at a roughly fixed height, so visible sky concentrates there when present, though this breaks down under tree canopy or in dense urban canyons). Two per-pixel HSV rules combine into a sky mask: bright + low-saturation (catches overcast/white sky) OR blue-hued + bright (catches clear sky) — a single blue-only rule would have missed every overcast photo. Two outputs, not one, because "how much sky" and "what kind of sky" are different claims: `sky_fraction` (proportion of the top third classified as sky, an openness/urban-density proxy) and `sky_blue_ratio` (of the sky pixels found, what fraction are blue rather than grey — the actual climate signal, expected to mainly separate South Africa/Australia-type intense-blue-sky scenes from everywhere else, per direct GeoGuessr playing experience). No shape or contiguity check — the mask is purely per-pixel, so scattered sky glimpsed through tree gaps scores identically to one open patch of equal pixel count.

**HOG (Histogram of Oriented Gradients).** Grayscale (shape/edge structure, independent of color, so no overlap with HSV). Resized to 128×128 first, since running HOG at native 640×640 would produce thousands of features — disproportionate to every other feature family. Settled on `pixels_per_cell=(32,32)`, `cells_per_block=(1,1)`, 9 orientation bins → a 4×4 spatial grid, 144 raw features. An initial attempt at a single global block (36 features) was rejected on the reasoning that it collapses the entire image into one "which direction do most edges point" number, destroying the spatial arrangement that's actually supposed to separate dense urban grids from open rural scenes — the 4×4 grid was chosen specifically to keep some of that spatial information while still staying far below full-resolution HOG's feature count.

**HOG dimensionality reduction.** 144 raw HOG features were compressed via PCA rather than hand-coarsened further, on the reasoning that PCA adapts to the actual variance structure of the data instead of applying a fixed, uniform blur the way a bigger cell size would. LDA was considered as a supervised alternative (optimizes for class separation directly via Fisher's criterion, rather than raw variance) but rejected for this baseline since it assumes roughly Gaussian, similar-covariance class distributions — an assumption with no verified basis for ~100 country classes on hand-engineered features. PCA's component count was chosen via a scree plot / cumulative explained variance read from a **train-only** full-component fit (the Kaiser criterion found 0 components with eigenvalue > 1, so cumulative variance against an 80% threshold was used instead), landing on 30 components. The **fit** step (`PCA(n_components=...).fit_transform`) runs only on the training split; the **transform** step reuses that exact fitted object on validation/test data — no separate fit, to avoid leaking test information into the compression itself. This same fit/transform split pattern, once established here, was reused for every other fitted transformation downstream (scaling, LBP's own PCA).

**LBP (Local Binary Patterns).** Grayscale, uniform method (`P=8, R=1`), which collapses 256 raw per-pixel pattern codes down to 10 meaningful "uniform" categories — most real texture signal concentrates there, the rest is largely noise. Computed on a 2×2 spatial grid (not a single global histogram) to preserve rough top/bottom, left/right texture location — asphalt at the bottom vs. foliage at the top is a real signal a single global histogram would erase. Produced 40 raw features (10 bins × 4 quadrants), later PCA-reduced the same train-fit/test-transform way as HOG, landing on components read off its own scree plot.

**Stopping point for hand-engineered feature *complexity*, explicitly acknowledged mid-project.** A planned "dominant line orientation via Hough transform" feature was scoped in detail (weighted angle histogram + line-density count) but ultimately dropped — it would have been a refinement on the same structural information HOG already captures, and continuing to add increasingly elaborate hand-built descriptors was recognized as running past the actual learning goal of this phase (understanding how images become numeric features), not toward it. Three additional simple, same-cost-tier features were added instead, and feature engineering was closed out there:

- **Ground color** (bottom third of frame) — mirrors the sky feature's positional-region approach for the ground plane instead of the sky: mean hue/saturation/value plus a "grey/paved" fraction, aimed at picking up road surface material (grey asphalt vs. red laterite vs. dirt), a cue used directly by human GeoGuessr players.
- **Global brightness/contrast** — mean and standard deviation of the grayscale image, a proxy for overall atmospheric conditions (haze/overcast vs. clear/harsh light) that nothing else in the feature set targeted.
- **Edge density** (via Canny), split into top-half/bottom-half/total — distinct from HOG, which captures edge *direction*; this captures edge *amount and location*, a second angle on the same open-road-vs.-dense-city question.

**Vegetation index, revisited with more skepticism.** Before building it, direct experience playing GeoGuessr raised the objection that most countries show substantial vegetation in street-level photos regardless of identity, so this feature was expected to separate only climate extremes (desert nations, dense city-states, tropical nations) rather than carry broad signal across the full class set — built anyway on the basis that it costs little and the actual answer would come from feature importance once a baseline model existed, not from theory alone.

## Data quality checks

Modeled after tabular-data cleaning steps, translated to images: corruption (`cv2.imread` returning `None`), channel consistency (confirming every image is 3-channel — it is, uniformly, which later surfaces as a legitimately constant feature rather than a bug), and duplicate detection via perceptual hashing (`imagehash.phash`, chosen over exact pixel hashing so near-identical re-compressed frames are still caught) — the dataset came back clean on all three (0 corrupted, 0 duplicates, 100% 3-channel).

**Two real bugs surfaced and were fixed during this phase, both worth recording as lessons rather than just fixes:**
1. **Namibia's ISO code is the literal string `"NA"`.** Reading the saved CSV back with plain `pd.read_csv` silently converted every Namibian (and some neighboring-country) `country_code`/`coord_country_code` entry into an actual missing value, since `"NA"` is in pandas' default null-string list. Fixed by reading with `keep_default_na=False, na_values=[]`.
2. **OpenCV's Hue channel only spans 0–179, not 0–255.** The original HSV histogram function used a `[0, 256]` range uniformly across all three channels; the top four Hue bins (12–15) were asking for a numeric range that cannot physically occur in OpenCV's Hue encoding, producing identical (zero) values across every image and surfacing as `NaN`/constant-feature warnings in later feature-importance checks. Fixed by giving the Hue channel a distinct `[0, 180]` range, and every HSV feature was re-extracted afterward.

## Correlation analysis and feature selection

Pairwise correlation (threshold 0.8) surfaced redundancy in three places: within adjacent HSV histogram bins (a real pixel-distribution effect — shadow or vivid-color pixel mass spreads across several neighboring bins together, not into one bin discretely), within adjacent LBP bins (same underlying mechanism), and between `vari_mean`/`vari_std` (driven by the same underlying vegetation-mixture variable). A manually curated set of the most redundant columns was dropped outright.

Since `country` is categorical with no numeric ordering, Pearson and Kendall — both requiring numeric/ordinal variables — don't apply to feature-vs-target relationships here (Pearson remained the right and only tool for the feature-vs-feature multicollinearity check above). The correct feature-vs-categorical-target tools are the **ANOVA F-test** (`f_classif`, detects differences in class means) and **Mutual Information** (`mutual_info_classif`, detects any statistical dependency regardless of shape, catching relationships ANOVA's mean-based test could miss). Both were computed, and features falling in the bottom 20th percentile of *both* rankings simultaneously were dropped — requiring agreement between two differently-shaped metrics before removing a feature, rather than trusting either alone.

## Skewness, outliers, and scaling

Skewness (`|skew| > 2`) flagged 23 features, concentrated in HSV bins near the histogram's low-density tails, plus `vari_mean` and `lbp_pca_2`. Before applying `log1p` (which requires non-negative input), each flagged column was checked for negative values: `vari_mean` (meaningfully signed — positive indicates vegetation-leaning) was log-transformed with sign preserved (`sign(x) * log1p(|x|)`); `lbp_pca_2`, a PCA axis with no inherently meaningful sign or scale, was excluded from log-transformation entirely rather than forced through a transform that doesn't apply to it; every other flagged column received a standard `log1p`.

Outliers were handled via winsorization — clipping each continuous feature to its 1st/99th percentile — with bounds computed from the training split only and the identical bounds then applied to the test split, consistent with the fit-train/apply-test rule established during PCA. `StandardScaler` followed the same rule: fit on training features only, applied to test via `.transform()`.

## Cross-validation decision

A more rigorous reference pipeline (from unrelated prior work) refits every preprocessing step — imputation, feature engineering, winsorization, feature selection, scaling — inside each individual cross-validation fold, to prevent any fold's validation rows from influencing a transformation used on that fold's own training rows. Given the scale of rebuilding required to retrofit that discipline here, this project explicitly accepts a lighter-weight, disclosed limitation instead: scaling and PCA were fit once on the full training split rather than refit per fold, and a single stratified train/validation split is used in place of k-fold cross-validation for this first-pass baseline. The held-out test set itself remains uncontaminated throughout (all fitted transforms were fit on training data only, before test data was ever touched) — the accepted leakage is confined to comparisons *within* the training data.

## Train/validation split and final feature set

A second `StratifiedShuffleSplit` (10%, stratified by `country`) carved a validation set out of the training split, separate from the test set held out earlier in the project. Validation-size reasoning: with a dataset this size, the number of examples per class matters more than the overall percentage held out — the binding constraint is the smallest country classes (as few as ~110 images after earlier filtering), not the dataset's total size. Final shapes: 86,471 training rows, 9,608 validation rows, 10,676 test rows, 80 feature columns after all cleaning and selection.

## Baseline models

One model per cell, evaluated with both standard accuracy and **top-3 accuracy** — motivated directly by how GeoGuessr scoring actually feels: guessing a neighboring, plausible country is a meaningfully different mistake than a wildly wrong one, and top-1 accuracy alone doesn't capture that distinction. Each model's validation predictions are also expanded into explicit `guess_1`/`guess_2`/`guess_3` columns alongside the true country, enabling direct inspection of confusion patterns (e.g., which specific countries get mixed up with which) rather than only a summary metric.

Trained so far, all at default or lightly-constrained settings appropriate for a first-pass check rather than a tuned final model: **Logistic Regression** (default), **Decision Tree** (default), **Random Forest** (`n_estimators=50, max_depth=15`, explicitly reduced from a heavier default to keep first-pass runtime reasonable — intended to be retuned later), **KNN** (flagged in advance as the model expected to underperform here, on the reasoning that Euclidean distance degrades in ~80-dimensional mixed-scale feature space — the "curse of dimensionality" — and KNN has no built-in way to weight more-informative features over less-informative ones the way RF/LR implicitly do), and **Naive Bayes** (flagged in advance as likely disadvantaged by its feature-independence assumption, which the correlation analysis above already showed is violated by several feature pairs).
  
## Tech stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| `pandas` / `numpy` | Data handling |
| `Pillow` / `opencv-python` | Image loading and processing |
| `reverse_geocoder` | Coordinate → country code |
| `pycountry` / `pycountry-convert` | Country code → name / continent |
| `plotly` | Exploratory visualizations |
| `kagglehub` | Dataset acquisition |
| `scikit-learn` (planned) | Classical baseline models |

## Acknowledgments

- Dataset: [sylshaw/streetview-by-country](https://www.kaggle.com/datasets/sylshaw/streetview-by-country/data)
- Structural inspiration for the metadata-extraction pipeline: [crypticsy — "Geo Guesser"](https://www.kaggle.com/code/crypticsy/geo-guesser) by Animesh Singh Basnet
- Learning progression and pedagogical approach: Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — [ageron/handson-ml3](https://github.com/ageron/handson-ml3)
- Classical feature-engineering approach grounded in Hays & Efros, *IM2GPS: Estimating Geographic Information from a Single Image* (2008)

## License / usage note

The source dataset's license is listed as "Unknown" on Kaggle. This project is personal and educational; it is not intended for commercial use or redistribution of the underlying imagery.
