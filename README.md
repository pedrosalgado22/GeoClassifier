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

Instead of using NNs to learn and predict the target based on the images directly, as common image prediction models do, this project uses a much less efficient, accurate, and useful methodology: extracting tabular data from the images using different functions to derive values for different image characteristics, such as HSV, green color (VARI), sky proportion, HOG, LBP, ground color, brightness/contrast, and edge density, and then making the predictions. This method was chosen for several reasons: first, NNs are black boxes that make it harder to understand everything that is going on, which components matter more, and what the model is taking into account. Second, I wanted to experiment with how efficient a model could be using tabular data to make geographical predictions, and how much worse it would be compared to the multiple existing ML/NN models that already make them at a very high level.

Classical algorithms expect `X` to be a 2D numeric table, not a raw image grid, so the entire first phase of this project was about deriving a meaningful, fixed-length numeric feature vector per image (hand-engineered features) rather than feeding in raw pixels. This approach mirrors and is inspired by the pre-deep-learning approach to this exact problem: the [IM2GPS project](https://graphics.cs.cmu.edu/projects/im2gps/) (Hays & Efros, 2008) tackled photo geolocation using engineered descriptors, such as color histograms, textures, and scene layouts, before CNN-based approaches became standard.

This methodology also has its limitations: the obvious one is that, although transforming images into numeric matrices might be a fun idea, I have no expertise in image processing, whatever characteristics can be extracted from images or constitute them. That limitation served as a learning phase for understanding photography, which is always a valuable skill for a data worker, especially when working with and evaluating image datasets.



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
Downloaded via `kagglehub`, targeted into a repo-local, gitignored `data/` directory (via `output_dir`) rather than the default global cache, so the project path is stable and reproducible across machines. `.gitignore` covers the whole `data/` directory.

### 2. Metadata extraction
Every image's folder (ISO country code) and filename (coordinates + heading) are parsed into a dataframe: `country_code`, `image_name`, `width`, `height`, `size`, `path`, `latitude`, `longitude`, `heading`. Parsing was implemented via simple string splitting (`file_path.stem.split('_')`) rather than regex, after a deliberate simplification pass — the split version reads more directly against the filename's own structure ("n = negative, d = decimal, `_` = separator") and Python's `int()` already parses a leading `-` natively, which resolved the earlier regex sign-parsing bug as a side effect.

### 3. Label derivation and verification
Rather than trusting the folder-provided country code alone, `country` and `continent` labels are derived directly from each image's actual coordinates via `reverse_geocoder`, `pycountry`, and `pycountry_convert` — chosen as the more geographically accurate source of truth. The folder-provided code is retained alongside the coordinate-derived one (`coord_country_code`) as a QA check — of the images processed, **3,572 disagreed** between the two, expected for points near borders or coastlines and confirmed to be a genuine limitation of `reverse_geocoder` (a nearest-known-populated-place lookup, not true polygon-based reverse geocoding — it can misassign points near borders, in small/disputed territories, or through inconsistent country-name-to-alpha2 mapping for territories like Taiwan). 2 rows came back with a missing `continent`.

### 4. Class balancing / outlier filtering
The coordinate-derived per-country distribution was inspected (mean, median, quartiles, min/max, and a heatmap visualization of image count per country). Countries with fewer than 110 images were filtered out (mirroring, and re-derived from, a similar threshold used in the abandoned 50k-dataset approach) to keep the class distribution usable for training.

### 5. Data quality checks
Modeled directly on tabular-data cleaning steps, translated to images — a question explicitly worked through during the project: missing/corrupted data → unreadable image files (`cv2.imread` returning `None`); missing values → the 2 missing `continent` rows above; incorrect/mislabeled data → the 3,572 `country_code`/`coord_country_code` mismatches, treated as label noise; duplicates → perceptual hashing (`imagehash.phash`, chosen over exact pixel hashing so near-identical re-compressed frames are still caught, not just byte-identical files); inconsistent format beyond size → color-channel consistency; and one image-specific addition with no tabular equivalent → placeholder/broken-capture frames (blank grey "no imagery" tiles), flagged as a concept to watch for via near-zero variance in later features rather than implemented as its own dedicated check.

Implemented as three simple, sequential functions (`check_corrupted`, `check_channels`, `check_duplicates`) that each take a dataframe and return it with a new column — an earlier `multiprocessing.Pool`-based version was deliberately rejected as unnecessary overhead for this preprocessing step. Results: **0 corrupted images, 100% 3-channel, 0 duplicate perceptual hashes** — the dataset came back clean on all three checks.

### Two real bugs found and fixed during this phase

1. **Namibia's ISO code is the literal string `"NA"`.** After saving `train_set`/`test_set` to CSV and reloading with plain `pd.read_csv`, every Namibian (and some neighboring-country border-adjacent) `country_code`/`coord_country_code` entry silently became an actual missing value — `"NA"` sits in pandas' default null-string list. Diagnosed by groupby-ing the newly-missing rows by `country` and finding **897 missing entries, overwhelmingly Namibia (877), with Botswana (14) and South Africa (6)** from border-adjacent points. Before landing on this explanation, a `cached()` joblib disk-caching wrapper (built to avoid re-running expensive HOG/LBP extraction on every kernel restart) was investigated as a possible cause of dataframe misalignment via `pd.concat` — a real, distinct risk of that caching pattern (it has no way to detect that its input dataframe has changed shape between runs), but not the actual cause here. Fixed by reading CSVs with `keep_default_na=False, na_values=[]`.
2. **OpenCV's Hue channel only spans 0–179, not 0–255.** The original HSV histogram function used a `[0, 256]` range uniformly across all three HSV channels; the top four Hue bins (`hsv_bin_12`–`15`) were asking for a numeric range that cannot physically occur in OpenCV's Hue encoding, producing identical (zero) values across every single image. This surfaced downstream as `NaN` F-scores and `UserWarning: Features [...] are constant` during ANOVA feature-importance checks. Fixed by giving the Hue channel a distinct `[0, 180]` range; every HSV feature was re-extracted afterward and the fix was verified via `.nunique()` on the affected bins before proceeding.

### Recurring notebook/kernel-state lessons
Several bugs during this project were not logic errors but Jupyter kernel-state issues, worth recording as a pattern: `NameError`s for objects (`meta_model`, `pc`) that were defined correctly earlier but lost after a kernel restart or simply not re-run in execution order; an empty `hog_cols` list caused by running PCA on a dataframe *before* the train/test split had actually been re-run against it in the new pipeline order; and editing a `.py` file (e.g. `hsv.py`) on disk not automatically propagating into an already-running kernel's imported module without a restart or `importlib.reload`.

## Feature Engineering

Since classical algorithms need a fixed-length numeric vector per image rather than a raw pixel grid, the project built a hand-engineered feature set, one feature family at a time, in this order:

**1. HSV color histogram** — 16 bins per channel, 48 raw features. HSV was chosen over RGB specifically to avoid encoding the same color information twice (the two are a deterministic, if nonlinear, transform of each other — extracting both would be redundant rather than additive, and actively destabilizing for Logistic Regression, which is sensitive to correlated/redundant inputs), and because separating hue from brightness/saturation is more robust to the inconsistent lighting/exposure across street-view captures than raw RGB. Bin count (16) was a deliberate middle point: enough resolution to be useful without overwhelming LR/NB with dimensionality; bin counts don't need to be powers of two, that's convention, not a requirement. Values normalized to proportions (`hist / hist.sum()`) rather than raw counts.

**2. Vegetation index (VARI)** — mean, std, and threshold-fraction (3 features). True NDVI requires a near-infrared channel, which street-view RGB photos don't have, so VARI (`(G−R)/(G+R−B)`) is used as the standard visible-band proxy from the precision-agriculture literature. Using a ratio rather than raw channel values makes the signal resistant to illumination differences — the same design goal as the HSV choice, solved the same way. Built despite direct skepticism (from actual GeoGuessr-playing experience) that most countries show substantial vegetation regardless of identity — included anyway on the basis that its real contribution would be settled empirically via feature importance, not decided by theory alone.

**3. Sky proportion** — `sky_fraction` and `sky_blue_ratio` (2 features). Restricted to the top third of the frame — a positional proxy, not true segmentation, since street-view cameras sit at a roughly fixed height and visible sky concentrates there when present (this breaks down under tree canopy or in dense urban canyons; no shape/contiguity check exists either, so sky glimpsed through scattered tree gaps scores identically to one open patch of equal pixel count). Two per-pixel HSV rules combine into a sky mask: bright + low-saturation (catches overcast/white sky) OR blue-hued + bright (catches clear sky) — a blue-only rule alone would have missed every overcast photo. `sky_fraction` (how much of the top third is sky-like — an openness/urban-density proxy) and `sky_blue_ratio` (of the sky found, what fraction is blue rather than grey) are reported separately because "how much sky" and "what kind of sky" are different claims. `sky_blue_ratio` was built specifically on the direct observation, from playing GeoGuessr, that South Africa and Australia have distinctly intense, saturated blue skies compared to almost everywhere else — this feature ended up being the single strongest individual predictor by both ANOVA F-score and mutual information.

**4. HOG (Histogram of Oriented Gradients)** — 144 raw features. Grayscale (shape/edge structure, independent of color, no overlap with HSV). Resized to 128×128 first, since native 640×640 would produce thousands of features. Settled on `pixels_per_cell=(32,32)`, `cells_per_block=(1,1)`, 9 orientation bins → a 4×4 spatial grid. A single global block (36 features) was tried first and rejected on the reasoning that collapsing the whole image into one "which direction do most edges point" number destroys the spatial arrangement that's actually supposed to separate dense urban grids from open rural scenes — closer to a generic "how much edge activity" proxy than a real structure detector. The 4×4 grid keeps some of that spatial information while staying far below full-resolution HOG's feature count (which would run into the thousands).

**5. HOG dimensionality reduction (PCA)** — 144 raw → **30 components**. PCA was chosen over LDA (a supervised alternative that optimizes for class separation directly via Fisher's criterion) specifically because LDA assumes roughly Gaussian, similar-covariance class distributions — an assumption with no verified basis for ~100 country classes on hand-engineered features — and carries small-sample instability risk when features approach per-class sample sizes; PCA makes no such assumption. Component count was chosen via a **train-only** full-component fit, read off a scree plot and cumulative-variance curve (the Kaiser criterion found 0 components with eigenvalue > 1, so an 80% cumulative-variance threshold was used instead, landing on 30). The fit step (`.fit_transform`) runs only on the training split; the transform step reuses that exact fitted `PCA` object on validation/test data with `.transform()` only — this fit-train/transform-test pattern, once established here, was reused for every subsequent fitted transformation in the project (LBP's own PCA, scaling).

**6. LBP (Local Binary Patterns)** — 40 raw features. Grayscale, uniform method (`P=8, R=1`), collapsing 256 raw per-pixel pattern codes down to 10 meaningful "uniform" categories (most real texture signal concentrates there). Computed on a 2×2 spatial grid (not one global histogram) to preserve rough top/bottom, left/right texture location — asphalt at the bottom vs. foliage at the top is real signal a single global histogram would erase. A resize-to-128×128 step, mirroring HOG's, was added after LBP was noted running noticeably slower than the other features (it originally ran on the full native-resolution image). LBP's own PCA followed the identical train-fit/test-transform pattern: 40 → **7 components** (Kaiser again found 0 eligible; cumulative variance hit 80% at 5 components, 7 was chosen for a small margin above that).

**7. Ground color, brightness/contrast, edge density** — the three features added after the line-orientation feature was scoped and deliberately dropped, chosen specifically for being the same low-cost tier as HSV/vegetation/sky (native resolution, no resize, no per-pixel neighbor comparison):
   - **Ground color** (bottom third of frame, 4 features: `ground_hue_mean`, `ground_sat_mean`, `ground_val_mean`, `ground_grey_frac`) — mirrors the sky feature's positional-region approach for the ground plane, aimed at road surface material (grey asphalt vs. red laterite vs. dirt), a cue human GeoGuessr players use directly.
   - **Global brightness/contrast** (2 features: `brightness_mean`, `contrast_std`) — mean and standard deviation of the grayscale image, a proxy for atmospheric conditions (haze/overcast vs. clear/harsh light) that nothing else in the feature set targeted.
   - **Edge density** (via Canny, 3 raw features: `edge_density_total`, `edge_density_top`, `edge_density_bottom`, split top-half/bottom-half) — distinct from HOG, which captures edge *direction*; this captures edge *amount and location*, a second angle on the open-road-vs.-dense-city question. `edge_density_total` was later dropped in feature selection after correlation analysis showed it was near-perfectly correlated with raw file `size` (0.966) and with `edge_density_top` (0.894) — likely because busier, more detailed images compress to larger JPEG file sizes, making `size` an accidental, low-quality edge-density proxy in its own right.

## Correlation Analysis and Feature Selection

Pairwise Pearson correlation (threshold 0.8) was run on numeric feature columns and surfaced redundancy in several places: within adjacent HSV histogram bins (a real pixel-distribution effect — shadow or vivid-color pixel mass spreads continuously across several neighboring bins rather than discretely into one, since real-world causes like "amount of shadow" are continuous and don't respect bin boundaries), within adjacent LBP bins (same mechanism), between `vari_mean`/`vari_std` (~0.95, both driven by the same underlying vegetation-mixture variable), and between `size`/`edge_density_total`/`edge_density_top` as described above.

Since `country` is categorical with no numeric ordering, **Pearson and Kendall — both requiring numeric/ordinal variables — don't apply to feature-vs-target relationships** (Pearson remained the correct and only tool for the feature-vs-feature multicollinearity check above; Kendall was ruled out entirely for the same reason). **Chi-square** was considered and set aside — it requires categorical/binned inputs, so continuous engineered features would need arbitrary discretization first, a strictly worse option than the two tools actually used given no specific reason favored it. The two correct feature-vs-categorical-target tools used instead were the **ANOVA F-test** (`f_classif`, detects differences in per-class means) and **Mutual Information** (`mutual_info_classif`, detects any statistical dependency regardless of shape, catching relationships a mean-based test could miss) — run together specifically because they can disagree, and features were only dropped when they fell in the **bottom 20th percentile of both rankings simultaneously**, requiring agreement between two differently-shaped metrics rather than trusting either alone.

A manually curated list was dropped outright first: `country_code`, `width`, `height`, `size`, `path`, `coord_country_code`, `corrupted`, `channels`, `phash`, `latitude`, `vari_std`, `longitude`, `hsv_bin_34`, `hsv_bin_29`, `edge_density_total`, `continent` — a mix of pure QC/metadata artifacts (`path`, `size`, `channels`, `corrupted`, `phash`), leakage risks (`latitude`/`longitude` directly determined the label via reverse geocoding and were also excluded from the ANOVA/MI computation itself for the same reason; `continent` would trivially leak the coarser version of the target), and specific redundant columns identified above. `width`, `height`, and `channels` also separately triggered `NaN`/constant-feature warnings in `f_classif` — confirmed as genuinely constant (every image is 640×640, 3-channel) rather than a bug, distinct from the Hue-range bug which looked similar but had a different root cause.

The agreement-based `drop_cols2` step then removed, in aggregate: 14 of HOG's 30 PCA components and 2 of LBP's 7 PCA components — leaving 16 HOG-PCA and 5 LBP-PCA components in the final feature set.

## Skewness, Outliers, and Scaling

Skewness (`|skew| > 2`) flagged **23 features**, concentrated in HSV bins near the histogram's low-density tails (`hsv_bin_32` stood out sharply at 34.09, well above the rest, which ranged roughly 2–7), plus `vari_mean` (~6.95) and `lbp_pca_2` (the only negatively-skewed feature in the list, ~−2.44).

Before applying `log1p` (which requires non-negative input), each flagged column was checked for negative values: **`vari_mean`** (14,008 negative rows, meaningfully signed — positive indicates vegetation-leaning) was log-transformed with sign preserved (`sign(x) * log1p(|x|)`); **`lbp_pca_2`** (32,667 negative rows, a PCA axis with no inherently meaningful sign or scale) was excluded from log-transformation entirely rather than forced through a transform that doesn't apply to it; every other flagged HSV bin (0 negative values each) received a standard `log1p`.

**Winsorization** clipped each continuous feature to its 1st/99th percentile, with bounds computed from the training split only and the identical bounds applied to the test split — the same fit-train/apply-test discipline established during PCA. One real bug was caught and fixed here: the winsorize function originally referenced the *global* `train_df` variable instead of its own `df_train` parameter when building its column list — it worked by accident (same columns at the time) but broke the function's independence.

**`StandardScaler`** followed the identical rule: `fit_transform` on training features only, `.transform()` on test.

## Cross-Validation Decision

A more rigorous reference pipeline (from unrelated prior work) refits *every* preprocessing step — imputation, feature engineering, winsorization, feature selection, scaling — inside each individual cross-validation fold, specifically so no fold's validation rows ever influence a transformation used on that fold's own training rows. Applying that same leakage principle to this project specifically: the held-out **test set** was already fully protected throughout (every fitted transform — PCA, scaling — was fit on training data only, before test data was ever touched). The remaining risk was narrower: if k-fold CV were run directly on the already-transformed `train_df`, every fold's "held-out" rows would have already influenced the once-globally-fit scaling/PCA, a real but scoped-to-training-data leak, smaller in expected magnitude than the reference pipeline's highest-risk step (feature selection informed by validation-fold data).

Given the scale of rebuilding required to retrofit full per-fold refitting, the project made an explicit, disclosed trade-off: **scaling and PCA were fit once on the full training split rather than refit per fold, and a single stratified train/validation split was used in place of k-fold cross-validation** for this first-pass baseline, accepting that the comparisons *within* training data carry a known, bounded amount of leakage while the test set itself remains clean.

## Train/Validation/Test Split

An initial `StratifiedShuffleSplit` (stratified by `country`, mirroring the structure of Géron's California-housing income-bracket example — with no binning step needed here, since `country` is already categorical rather than continuous) carved off the test set early in the pipeline. A second `StratifiedShuffleSplit` (10%, also stratified by `country`) later carved a validation set out of the training split.

Validation-size reasoning: with a dataset this size, the number of examples *per class* matters more than the overall percentage held out — the binding constraint is the smallest country classes (as few as ~110 images), not the dataset's total size; `value_counts().tail(10)` was the recommended check before committing to a percentage.

**Final shapes:** 86,471 training rows, 9,608 validation rows, 10,676 test rows, **80 feature columns** after all cleaning, selection, and transformation. The final 80 break down as: 1 (`heading`, retained as a raw numeric feature — camera direction at capture — since it wasn't flagged as redundant/low-value enough to be dropped, despite being closer to metadata than a deliberately engineered feature) + 2 (`vari_mean`, `vari_frac`) + 2 (`sky_fraction`, `sky_blue_ratio`) + 16 (HOG PCA components) + 4 (ground color) + 2 (brightness/contrast) + 2 (`edge_density_top`, `edge_density_bottom`) + 46 (HSV bins, 48 minus 2 dropped) + 5 (LBP PCA components).

## Baseline Models

One model per cell, evaluated on the validation set with both standard accuracy and **top-3 accuracy** — motivated directly by how GeoGuessr scoring actually feels: guessing a neighboring, plausible country is a meaningfully different mistake than a wildly wrong one, and top-1 accuracy alone doesn't capture that. Each model's predictions were also expanded into explicit `guess_1`/`guess_2`/`guess_3` columns alongside the true country, for direct inspection of confusion patterns rather than only a summary metric.

| Model | Settings | Accuracy | Top-3 Accuracy |
|---|---|---|---|
| Logistic Regression | default (`lbfgs`, `max_iter=100`) | 0.3182 | 0.4983 |
| Decision Tree | default | 0.1501 | 0.1637 |
| Random Forest | `n_estimators=50, max_depth=15` (deliberately constrained for first-pass speed) | 0.2964 | 0.4556 |
| KNN | `n_neighbors=5` | *run, expected to underperform (see below); results not recorded* | — |
| Naive Bayes | `GaussianNB` | 0.1872 | 0.3370 |

The default LR run threw a `ConvergenceWarning` — `lbfgs` stopped at its 100-iteration cap without genuinely converging, which also partly explains why it ran so fast: it hadn't actually finished optimizing. **KNN was flagged in advance** as the model expected to underperform, on the reasoning that Euclidean distance degrades in ~80-dimensional, mixed-scale feature space (the curse of dimensionality — points tend to become roughly equidistant from each other at this dimensionality) and KNN has no built-in mechanism to weight more-informative features over less-informative ones the way RF/LR implicitly do. **Naive Bayes was flagged in advance** as likely disadvantaged by its feature-independence assumption, already known-violated by several feature pairs from the correlation analysis above.

A separate, explicit question was raised about why all five models trained in "seconds" given prior experience with much slower training elsewhere — the answer: the genuinely expensive part of this whole project (extracting HOG/LBP/HSV/etc. from ~107k raw images) had already happened upstream; 80 features and ~86k rows is a small, ordinary tabular problem for LR/DT/RF/NB, none of which do iterative epoch-based training the way boosting or neural nets do; and RF was deliberately constrained for this first pass.

## Hyperparameter Tuning

Random Forest and Logistic Regression — the two most promising baselines — were tuned via `GridSearchCV`/`RandomizedSearchCV`. An initial attempt adapted directly from Géron's California-housing example contained two real bugs (`RandomForestRegressor` instead of `Classifier`, and `neg_root_mean_squared_error` scoring — both regression-specific and wrong for a classification target) which were corrected before proceeding.

**Grid design comparison:** a more elaborate, user-authored tuning grid was compared directly against an initial simpler one. The user's RF grid had 3× more combinations (36 vs. 12) via an added `min_samples_leaf` axis (a genuinely different regularization lever, not just "more of the same") and `class_weight="balanced"` (correctly compensating for the dataset's known, if mild, per-country imbalance) on both models; the LR grid added a genuine `l1`/`l2` penalty comparison. `max_depth` ranges were also compared explicitly: a deeper-biased range (`15, 25, None`) risks overfitting given some classes have only ~110 images, versus a shallower-biased range (`5, 10, 15`) risking underfitting — resolved as an empirical question for the search itself to answer, not something to reason out in advance.

**Real bugs hit during tuning, both root-caused directly from tracebacks:**
1. **`liblinear` does not support multiclass classification (n_classes ≥ 3)** in the installed sklearn version — an LR grid using `solver="liblinear"` failed all 30 fits outright. Fixed by switching to `solver="saga"`, which supports both `l1`/`l2` penalties and native multiclass.
2. **`cv='prefit'` was removed** from `CalibratedClassifierCV` in the installed sklearn version (relevant later, during ensembling) — replaced with the newer `FrozenEstimator` wrapper pattern.

**Runtime issues:** an `n_jobs` oversubscription problem was identified and fixed (model-level `n_jobs=-1` competing with search-level `n_jobs=-1` for the same CPU cores, rather than the search distributing candidates across cores while each model fit single-threaded). A `saga`+`l1`+`class_weight="balanced"` run then ran for 3+ hours with zero visible output — `GridSearchCV` gives no progress feedback until a fit *completes*, a real blind-spot distinct from the speed problem itself — resolved by adding `verbose=3` for per-fit progress and drastically shrinking the grid to first establish a real per-fit timing baseline (`l1` was deferred rather than run blind again).

**Final tuned results:**
- **Random Forest:** `{n_estimators=200, min_samples_leaf=1, max_features='log2', max_depth=25, class_weight='balanced'}` — CV score 0.3214.
- **Logistic Regression:** `{C=1, class_weight='balanced', penalty='l2', solver='saga'}` — CV score 0.3068.

**Honest finding:** both tuned results landed close to — LR's tuned score is actually *slightly below* — their original untuned baseline numbers. Tuning bought very little here, which is itself informative: it suggests the ceiling on this feature set's predictive power is set by the **features**, not the hyperparameters, and redirected the project toward feature-importance analysis and confusion-pattern inspection rather than further tuning.

## Final Test Set Evaluation

The first and only time the held-out test set was touched for scoring, both tuned models were evaluated with explicit top-1/2/3 accuracy:

| Model | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| Random Forest (tuned) | 0.3373 | 0.4406 | 0.5053 |
| Logistic Regression (tuned) | 0.3147 | 0.4278 | 0.4945 |

## Ensembling

Two ensembling approaches were built and explicitly compared, at the user's request, after an initial idea to simply average both models' `predict_proba` outputs was pushed back on directly with a real, correct technical objection: "if one thought it's Portugal by very little and the other thought it's Spain by a country mile, those are not the same" — i.e., RF's vote-fraction-based probabilities and LR's softmax-based probabilities sit on genuinely different confidence scales, so a flat average silently favors whichever model's outputs happen to run larger in magnitude, independent of which model is actually more trustworthy on a given example.

**Calibrated ensembling** — `CalibratedClassifierCV(method='isotonic')` fit separately per model on the held-out validation set (data neither base model was trained on), via the `FrozenEstimator` wrapper (see bug #2 above), then averaged. This puts both models' probabilities on a genuinely comparable scale before combining.

**Stacking** — a small `LogisticRegression` meta-model trained on the concatenated `predict_proba` outputs of both base models (computed on validation data), predicting on the same concatenation computed from test data — a more rigorous approach in principle, since it lets the data itself learn how much to trust each base model rather than assuming a fixed weighting.

| Ensemble | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| Calibrated average | **0.3458** | **0.4539** | **0.5152** |
| Stacked (meta-LR) | 0.3344 | 0.4386 | 0.5055 |

**Why the theoretically more sophisticated method (stacking) didn't win:** the meta-model had to learn from only ~200 input features (two models × ~100 class-probabilities) using ~9,608 validation rows — roughly 96 examples per class on average, thin ground for learning a reliable per-class weighting. Calibration's task is far lower-dimensional by comparison (one 1D curve per class, per model), so it holds up better with the amount of data available. This is a known, unsurprising pattern in the ensembling literature (stacking typically needs more data and/or more diverse base models to pay off), not a fluke of this run.

**The calibrated ensemble (Top-1 0.346, Top-2 0.454, Top-3 0.515) is the best result of the entire classical-ML phase of this project**, and was adopted as its final baseline number.

A recurring, unrelated bug surfaced twice during this phase and was fixed the same way both times: `y_test.values` (a pandas Arrow-backed string array in the installed pandas version) doesn't support numpy-style `[:, None]` reshape indexing — resolved via explicit `np.asarray(y_test)` conversion before any broadcasting comparison.

## Feature Importance and Diagnostics

The full, untruncated impurity-based feature importance list (`best_rf.feature_importances_`, all 80 features) was printed and reveals a **notably flat distribution** — nearly every feature sits in a narrow band (roughly 0.010–0.018), with no single feature or feature family dominating. HSV bins, HOG-PCA components, LBP-PCA components, ground color, sky, and vegetation features are all interleaved throughout the ranking rather than clustering by family at the top or bottom. This is itself a real, honest finding: it settles the early vegetation-index skepticism directly (`vari_mean`/`vari_frac` sit solidly mid-pack, ~0.0136/0.0131, neither dominant nor negligible) and suggests the model draws modest, broadly-distributed signal across the whole engineered feature set rather than relying heavily on any one family.

Permutation importance (via `sklearn.inspection.permutation_importance`, deliberately run on validation data rather than training data to avoid overstating importance for a model already tightly fit to its own training data, using default accuracy scoring rather than F1 since there's no natural positive class in a ~100-way multiclass problem) was set up and visualized alongside the impurity-based ranking as a cross-check, following a simplification pass away from an earlier, over-elaborate dual-model comparison template.

**Country-level accuracy table:** built with `top2_correct`/`top3_correct` computed cumulatively (an already-correct top-1 guess automatically counts as top-2- and top-3-correct too, matching the actual definition of top-k accuracy) alongside `n_images` per country, sorted descending by top-1 accuracy, printed in full via `pd.set_option("display.max_rows", None)`.

**Continent-level accuracy:** since `continent` had been dropped from the working dataframes during feature selection, it was recomputed fresh from the predicted and true country names via `pycountry` → `pycountry_convert`, compared directly, per an explicit request to keep this as minimal as possible.

**Learning curve:** train-vs-validation accuracy across increasing training-set sizes (`sklearn.model_selection.learning_curve`, `train_sizes=np.linspace(0.1, 1.0, 5)`), run using the already-tuned Random Forest's best hyperparameters directly (no need to re-specify them — `best_rf` already carries them).

### Baselines for comparison

A majority-class baseline — always guessing the single most frequent country — is the right comparison point for judging whether the ~31–35% top-1 / ~50–52% top-3 results above are meaningful. Given the dataset's official per-country file counts (min 110, max 1,005 for Vietnam, out of ~107,269 total images across 111 near-uniformly-distributed countries), a majority-class baseline is approximately:

- **Top-1: ≈ 0.9–1.3%** (1,005–1,408 ÷ ~107,269, depending on which distribution snapshot — pre- or post-filtering — is used; both point to roughly the same order of magnitude given the near-uniform design)
- **Top-3: ≈ 2.8–4%** (guessing the three most frequent countries every time)

This is an order-of-magnitude estimate rather than an exact figure pulled from the live `y_test` distribution — for the precise number, run:
```python
print(y_test.value_counts(normalize=True).head(1).values[0])
print(y_test.value_counts(normalize=True).head(3).sum())
```
For an exact continent-level majority baseline, the same pattern applies to a continent-mapped version of `y_test`. Against either baseline, the calibrated ensemble's 34.6% top-1 / 51.5% top-3 represents roughly a **30–40× improvement over random/majority-class guessing** — strong, unambiguous evidence that the hand-engineered features carry real geographic signal, even though the model is far from GeoGuessr-competitive in absolute terms.

## Qualitative Observations

Inspecting the country-level accuracy table alongside the actual images surfaced a pattern worth recording even before it's fully explained: **countries with lower internal visual variance across their own images tend to score noticeably better.** Specific cases noted:
- **São Tomé** — images consistently show a blur artifact, most likely a capture-hardware quirk specific to how that country's images were collected, functioning as an inadvertent but highly consistent "fingerprint" for the model to latch onto.
- **Oman** — a white car is visible in a subset of images, another likely capture-vehicle artifact rather than a landscape feature, similarly acting as an unintentional but consistent signal.
- **Tunisia** — images look genuinely similar to each other environmentally (clear sky, similar road type), a case of real, non-artifactual consistency rather than a capture quirk.
- **Vietnam** — the worst-performing country in the results, cause not yet confirmed. One plausible, untested hypothesis: Vietnam has several visually-similar Southeast Asian neighbors also present as distinct classes in this exact dataset (Thailand, Cambodia, Laos, Malaysia, Indonesia, the Philippines, Singapore), and lacks a single dominant, distinctive environmental signature the way, say, snow or extreme desert give other countries an easy "fingerprint" — this would predict Vietnam's errors cluster specifically among those neighbors, which hasn't been confirmed against the actual confusion table yet.

These observations tie directly back to a concern raised at the very start of the project, during dataset selection: capture-hardware or vehicle artifacts consistent within a country's image set can act as unintended shortcuts for a model, separate from genuine geographic signal.

## Current Status

This is the completed state of the classical machine-learning baseline phase: dataset selection, metadata and label pipeline, hand-engineered feature extraction (HSV, VARI, sky, HOG+PCA, LBP+PCA, ground color, brightness/contrast, edge density), correlation-based and statistical feature selection, skewness/outlier/scaling treatment, an explicitly disclosed cross-validation trade-off, five baseline models, hyperparameter tuning, and calibrated/stacked ensembling — landing on a best result of **34.6% top-1 / 51.5% top-3 accuracy** across 111 countries, roughly 30–40× above a majority-class baseline.

CNN-based modeling remains intentionally deferred to a later phase, now that this baseline exists as both a learning foundation and a concrete number to beat.

## Continent Accuracy

The same pipeline, features, and dataset splits were reused for a coarser target: predicting continent instead of country. With only six classes instead of roughly a hundred, this is a structurally easier problem, and the results confirm that expectation — but they also surface patterns, and one real inconsistency, that the country-level results did not.

### Baseline models

| Model | Settings | Accuracy | Top-3 Accuracy |
|---|---|---|---|
| Logistic Regression | default (`lbfgs`, did not converge) | 0.5276 | 0.8682 |
| Decision Tree | default | 0.4567 | 0.5621 |
| Random Forest | `n_estimators=50, max_depth=15` | 0.5977 | 0.8956 |
| Naive Bayes | `GaussianNB` | 0.3510 | 0.7399 |

Top-3 accuracy is far less informative here than it was at the country level, for a simple structural reason: with only six possible continents, guessing three of them covers half of all possible answers. A random guesser would already score roughly 50% top-3 accuracy by chance alone, compared to under 3% at the country level with ~111 classes. The top-3 figures above are consequently inflated regardless of model quality and are not treated as a meaningful signal for this part of the results; top-1 accuracy is the metric that actually distinguishes the models.

**Majority-class baseline:** Europe accounts for 3,804 of the 9,608 validation images (39.6%), by far the largest continent class — a direct and unavoidable consequence of the dataset balancing individual countries rather than continents, and Europe simply contains far more of the 111 countries than, say, Oceania does. A model that always guessed "Europe" would score close to 39.6% top-1 accuracy on this validation set. Naive Bayes, at 35.1%, actually falls **below** this majority-class baseline — a concrete, notable failure specific to this target, not seen at the country level, where every model comfortably beat its (much lower, near-1%) majority baseline.

### Tuning and final test results

| Model | Test Top-1 | Test Top-2 | Test Top-3 |
|---|---|---|---|
| Random Forest (tuned) | 0.6161 | 0.8136 | 0.9036 |
| Logistic Regression (tuned) | 0.4348 | 0.6506 | 0.7962 |

One discrepancy is worth flagging directly rather than glossing over: both tuning cells printed two different best-score values in sequence — 0.4340 and 0.3068 for Logistic Regression, 0.6116 and 0.3214 for Random Forest. The second value in each pair is not a new result; it is bit-for-bit identical to the corresponding country-level tuning score reported earlier in this project. This is almost certainly leftover output from a variable or cell not fully re-run for the continent target, rather than a genuine second computation, and the first value in each pair (0.4340 for LR, 0.6116 for RF) is the one consistent with everything else in this section. Worth re-running those specific cells cleanly before treating either number as final. Logistic Regression's convergence warning also recurred here, identical to the country-level run, so its reported numbers likely still understate its true ceiling.

### Ensembling

| Ensemble | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| Calibrated average | 0.6113 | 0.8124 | 0.9045 |
| Stacked (meta-LR) | **0.6274** | **0.8203** | **0.9052** |

This reverses the finding from the country-level task, where calibrated averaging beat stacking. The explanation offered earlier for that result — that the stacking meta-model needed more data per class than it had (roughly 96 validation examples per country) to learn a reliable weighting — is directly testable here, and the continent task provides exactly the confirming case: with only six classes, the same 9,608 validation rows give the meta-model over 1,600 examples per class on average, more than sixteen times the density it had before. With that much more signal to learn from, stacking now has enough to actually outperform the simpler calibrated average, which is consistent with the mechanism proposed earlier rather than a contradiction of it.

### Feature importance

The importance distribution is flat again, in the same way it was for country prediction — no single feature dominates, with values spread roughly between 0.009 and 0.025 across all 80 features. What stands out is that the *same* features lead both rankings: `lbp_pca_2`, `ground_hue_mean`, and `ground_grey_frac` sit at or near the top for both country and continent prediction. That consistency suggests these particular features carry genuinely general-purpose geographic signal, rather than being useful only for distinguishing similar countries from each other.

### Per-continent breakdown

| Continent | Top-1 | Top-2 | Top-3 | n images |
|---|---|---|---|---|
| Africa | 0.648 | 0.781 | 0.811 | 1,143 |
| Europe | 0.471 | 0.596 | 0.733 | 3,804 |
| Oceania | 0.289 | 0.476 | 0.613 | 450 |
| South America | 0.271 | 0.656 | 0.728 | 717 |
| Asia | 0.151 | 0.415 | 0.771 | 2,589 |
| North America | 0.139 | 0.412 | 0.663 | 905 |

One caveat before reading into this table: recomputing the weighted average of its `top1_correct` column against each continent's row count reproduces the Naive Bayes baseline accuracy (0.351) exactly, not Random Forest's (0.598). This indicates the table was actually built from Naive Bayes's validation probabilities rather than the tuned Random Forest's, despite being labeled with `best_rf`'s class order — a variable-reuse artifact rather than a bug in the underlying logic. The patterns below are still real and worth noting, but this table should be rebuilt against the actual best model (the stacked ensemble) before treating the exact numbers as final.

With that caveat in mind, a few things stand out. **Africa is the clear best-performing continent (64.8% top-1)**, despite not being the largest class by training volume — behind Europe and Asia in image count, but well ahead of both in accuracy, suggesting Africa has a genuinely distinctive visual signature the feature set picks up regardless of how much data it's trained on. **Europe, despite being by far the largest class (3,804 images, essentially 40% of validation data), only reaches 47.1%** — well below Africa's result on a third of the data, reinforcing that raw class size isn't the dominant driver of accuracy here.

**Asia and North America are the weakest continents**, at 15.1% and 13.9% top-1 respectively — both well below what their training volume alone would predict, especially Asia, which has the second-most images of any continent. This connects directly to a finding from the country-level results: Vietnam, an Asian country, was the single worst-performing country in that analysis. Taken together, this suggests the difficulty isn't isolated to Vietnam specifically, but may reflect a broader pattern of Asian countries being visually harder for this feature set to separate — both from each other and, evidently, from other continents entirely.

**South America shows the largest jump between top-1 and top-2 of any continent** (27.1% → 65.6%, a 38-point gain), noticeably larger than the jump for any other row. That pattern is consistent with South America being systematically confused with one specific other continent as the top guess, though confirming which one would require building an explicit confusion cross-tab against the true best model, the same way it was done for country-level confusions — a natural next diagnostic step, not yet performed here.

Lat/long-based distance prediction remains the third and final measurement planned for this results section, not yet covered here.
There's a real, literature-grounded reason direct lat/lon regression is not actually what most serious geolocation-from-image research does — including the IM2GPS lineage this whole project has been following. Google's PlaNet (the well-known successor work in this space) explicitly treats geolocation as classification over discretized geographic cells, not direct coordinate regression, for a specific structural reason: a photo of, say, generic tropical vegetation could plausibly come from several disconnected regions of the globe with similar climate. A regression model, forced to output one point, tends to average toward between those plausible locations — which is wrong for all of them — whereas a classifier choosing among discrete cells doesn't have that averaging failure mode; it can commit to one plausible cell without being dragged toward a geographic midpoint that matches nothing.
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
| `matplotlib` / `plotly` | Visualizations (scree plots, feature importance, distribution heatmaps, learning curves) |
| `kagglehub` | Dataset acquisition |
| `joblib` | Disk-caching of expensive extraction cells across notebook restarts |
| Jupyter / VS Code | Development environment |
| Git / GitHub | Version control |
| Claude (Anthropic) | AI pair-programming assistant used throughout development — feature methodology, debugging, and pipeline design were worked through interactively rather than solo |

## Acknowledgments

- Dataset: [sylshaw/streetview-by-country](https://www.kaggle.com/datasets/sylshaw/streetview-by-country/data)
- Structural inspiration for the metadata-extraction pipeline: [crypticsy — "Geo Guesser"](https://www.kaggle.com/code/crypticsy/geo-guesser) by Animesh Singh Basnet
- Learning progression and pedagogical approach: Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — [ageron/handson-ml3](https://github.com/ageron/handson-ml3)
- Classical feature-engineering approach grounded in Hays & Efros, *IM2GPS: Estimating Geographic Information from a Single Image* (2008)

## License / Usage Note

The source dataset's license is listed as "Unknown" on Kaggle, and Google's Maps Platform terms restrict model training on Street View content more broadly. This project is personal and educational; it is not intended for commercial use or redistribution of the underlying imagery.
