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
