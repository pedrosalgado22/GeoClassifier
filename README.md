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

Instead of using NNs to learn and predict the target based on the images directly, as common image prediction models do, this project uses a much less efficient, accurate, and useful methodology: extracting tabular data from the images using different functions to derive values for different image characteristics, such as HSV, green color (VARI), sky proportion, HOG, LBP, ground color, brightness/contrast, and edge density, and then making the predictions. This method was chosen for several reasons: first, NNs are black boxes that make it harder to understand everything that is going on, which is worse for learning. Second, I wanted to experiment with how efficient a model could be using tabular data to make geographical predictions, and how much worse it would be compared to the multiple existing ML/NN models that already make them at a very high level.

Classical algorithms expect `X` to be a 2D numeric table, not a raw image grid, so the entire first phase of this project was about deriving a meaningful, fixed-length numeric feature vector per image (hand-engineered features) rather than feeding in raw pixels. This approach mirrors and is inspired by the pre-deep-learning approach to this exact problem: the [IM2GPS project](https://graphics.cs.cmu.edu/projects/im2gps/) (Hays & Efros, 2008) tackled photo geolocation using engineered descriptors, such as color histograms, textures, and scene layouts, before CNN-based approaches became standard.

This methodology also has its many limitations: the obvious one is that, although transforming images into numeric matrices might be a fun idea, I have no expertise in image processing, whatever characteristics can be extracted from images or constitute them. That limitation served as a learning phase for understanding what constitutes an image, which is always a valuable skill for a data worker, especially when working with and evaluating photographic datasets.



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
`country` and `continent` are derived from each image's coordinates using `reverse_geocoder`, `pycountry` and `pycountry_convert`. The folder code is kept as `coord_country_code` for comparison. **3,572 images disagreed**, expected near borders and coastlines. Not super relevant for final results.

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

Implemented as three sequential functions (`check_corrupted`, `check_channels`, `check_duplicates`), each taking a dataframe and returning it with a new column.

**Results:** 0 corrupted images, 100% 3-channel, 0 duplicate hashes.

**Confidence:** 98% that no main steps or figures were lost or altered.

### Two real bugs found and fixed during this phase

1. **Namibia's ISO code is the literal string `"NA"`.**  Every Namibian (and some neighboring-country border-adjacent) `country_code`/`coord_country_code` entry silently became an actual missing value — `"NA"` sits in pandas' default null-string list. Diagnosed by groupby-ing the newly-missing rows by `country` and finding **897 missing entries, overwhelmingly Namibia (877), with Botswana (14) and South Africa (6)** from border-adjacent points.
2. **OpenCV's Hue channel only spans 0–179, not 0–255.** The original HSV histogram function used a `[0, 256]` range uniformly across all three HSV channels; the top four Hue bins (`hsv_bin_12`–`15`) were asking for a numeric range that cannot physically occur in OpenCV's Hue encoding, producing identical (zero) values across every single image. This surfaced downstream as `NaN` F-scores and `UserWarning: Features [...] are constant` during ANOVA feature-importance checks.

## Feature Engineering
### Feature engineering (7 families)

**1. HSV color histogram** — 16 bins/channel, 48 features. HSV over RGB to avoid redundant color info (RGB↔HSV is a deterministic transform, so using both would just add correlated noise) and because hue/brightness/saturation separate more cleanly under inconsistent street-view lighting. Normalized to proportions.

**2. Vegetation index (VARI)** — mean, std, threshold-fraction (3 features). `(G−R)/(G+R−B)`, the RGB-only proxy for NDVI (no infrared channel available). Kept despite suspecting most countries have "enough" vegetation to make this weak — left to feature importance to decide, not assumed.

**3. Sky features** — `sky_fraction`, `sky_blue_ratio` (2 features). Top third of frame only (position proxy, not segmentation — breaks under canopy/urban canyons). Sky mask = bright+low-saturation (overcast) OR blue+bright (clear). `sky_blue_ratio` targets the distinctly saturated blue skies of places like South Africa/Australia — turned out to be the single strongest predictor (ANOVA F-score, mutual information).

**4. HOG (edge structure)** — 144 features. Grayscale, resized to 128×128, `pixels_per_cell=(32,32)`, 9 orientation bins → 4×4 spatial grid. A single global block was tried and rejected — it collapses to "which way do edges point overall," losing the spatial layout needed to tell dense urban grids from open rural scenes.

**5. HOG → PCA** — 144 → 30 components. PCA over LDA: LDA assumes Gaussian, equal-covariance classes (unverified for ~100 countries) and is unstable with small per-class samples. Component count set via train-only scree/cumulative-variance plot (Kaiser criterion gave 0 useful components, so used 80% variance threshold → 30).

**6. LBP (texture)** — 40 features. Grayscale, uniform method (`P=8, R=1`), 256 raw codes → 10 uniform bins. Computed on a 2×2 grid, not global, to keep rough top/bottom texture location (foliage vs. asphalt). Resized to 128×128 after it ran slow at full resolution. Own PCA: 40 → 7 components, same train-fit/test-transform pattern as HOG's.

**7. Ground, brightness/contrast, edge density** — added as a cheap final tier (native resolution, no resizing):
- **Ground color** (4 features, bottom third): hue/sat/val means + grey-fraction — road surface material (asphalt vs. laterite vs. dirt).
- **Brightness/contrast** (2 features): mean + std of grayscale — atmospheric proxy (haze/overcast vs. clear).
- **Edge density** (3 features, via Canny, split top/bottom): edge *amount/location*, complementing HOG's edge *direction*. `edge_density_total` later dropped — found near-perfectly correlated with file `size` (0.966) and `edge_density_top` (0.894), since busier images just compress larger.


## Correlation Analysis and Feature Selection

Pairwise Pearson correlation (threshold 0.8) was run on numeric feature columns and surfaced redundancy in several places: within adjacent HSV histogram bins, within adjacent LBP bins (same mechanism), between `vari_mean`/`vari_std` (~0.95, both driven by the same underlying vegetation-mixture variable), and between `size`/`edge_density_total`/`edge_density_top` as described above.

Since `country` is categorical with no numeric ordering, **Pearson and Kendall don't apply to feature-vs-target relationships** (Pearson remained for the feature-vs-feature multicollinearity check above). **Chi-square** was considered and set aside as it requires categorical/binned inputs. The two used feature-vs-categorical-target tools used instead were the **ANOVA F-test** (`f_classif`, detects differences in per-class means) and **Mutual Information** (`mutual_info_classif`, detects any statistical dependency regardless of shape, catching relationships a mean-based test could miss) run together since they can disagree, and features were only dropped when they fell in the **bottom 20th percentile of both rankings simultaneously**, requiring agreement between two differently-shaped metrics rather than trusting either alone.

A manually curated list was dropped outright first: `country_code`, `width`, `height`, `size`, `path`, `coord_country_code`, `corrupted`, `channels`, `phash`, `latitude`, `vari_std`, `longitude`, `hsv_bin_34`, `hsv_bin_29`, `edge_density_total`, and `continent`, with a mix of pure QC/metadata artifacts, leakage risks and specific redundant columns identified above.

## Skewness, Outliers, and Scaling

Skewness (`|skew| > 2`) flagged **23 features**, concentrated in HSV bins near the histogram's low-density tails.

Before applying `log1p` (which requires non-negative input), each flagged column was checked for negative values: **`vari_mean`** was log-transformed with sign preserved (`sign(x) * log1p(|x|)`);

**Winsorization** clipped each continuous feature to its 1st/99th percentile, with bounds computed from the training split only and the identical bounds applied to the test split.

**`StandardScaler`** followed the identical rule: `fit_transform` on training features only, `.transform()` on test.

## Cross-Validation

**Scaling and PCA were fit once on the full training split rather than refit per fold, and a single stratified train/validation split was used in place of k-fold cross-validation** for this first-pass baseline, accepting that the comparisons *within* training data carry a known, bounded amount of leakage while the test set itself remains clean.

## Train/Validation/Test Split

**Final shapes:** 86,471 training rows, 9,608 validation rows, 10,676 test rows, **80 feature columns** after all cleaning, selection, and transformation.

## Baseline Models

One model per cell, evaluated on the validation set with both standard accuracy and **top-3 accuracy**, motivated directly by how GeoGuessr scoring feels: guessing a neighboring, plausible country is a meaningfully different mistake than a wildly wrong one, and top-1 accuracy alone doesn't capture that. Each model's predictions were also expanded into explicit `guess_1`/`guess_2`/`guess_3` columns alongside the true country, for direct inspection of confusion patterns rather than only a summary metric.

| Model | Settings | Accuracy | Top-3 Accuracy |
|---|---|---|---|
| Logistic Regression | default (`lbfgs`, `max_iter=100`) | 0.3182 | 0.4983 |
| Decision Tree | default | 0.1501 | 0.1637 |
| Random Forest | `n_estimators=50, max_depth=15` (deliberately constrained for first-pass speed) | 0.2964 | 0.4556 |
| KNN | `n_neighbors=5` | *run, expected to underperform; results not recorded* | — |
| Naive Bayes | `GaussianNB` | 0.1872 | 0.3370 |

**KNN was flagged in advance** as the model expected to underperform, on the reasoning that Euclidean distance degrades in ~80-dimensional, mixed-scale feature space (the curse of dimensionality: points tend to become roughly equidistant from each other at this dimensionality) and KNN has no built-in mechanism to weight more-informative features over less-informative ones the way RF/LR implicitly do. **Naive Bayes was flagged in advance** as likely disadvantaged by its feature-independence assumption, already known-violated by several feature pairs from the correlation analysis above.

## Hyperparameter Tuning

Random Forest and Logistic Regression — the two most promising baselines — were tuned via `GridSearchCV`/`RandomizedSearchCV`.

**Final tuned results:**
- **Random Forest:** `{n_estimators=200, min_samples_leaf=1, max_features='log2', max_depth=25, class_weight='balanced'}` — CV score 0.3214.
- **Logistic Regression:** `{C=1, class_weight='balanced', penalty='l2', solver='saga'}` — CV score 0.3068.

Tuning bought very little here, which  suggests the ceiling on this feature set's predictive power is mostly set by the **features**, not the hyperparameters, and redirected the project toward feature-importance analysis and confusion-pattern inspection rather than further tuning.

## Final Test Set Evaluation

The first and only time the held-out test set was touched for scoring, both tuned models were evaluated with explicit top-1/2/3 accuracy:

| Model | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| Random Forest (tuned) | 0.3373 | 0.4406 | 0.5053 |
| Logistic Regression (tuned) | 0.3147 | 0.4278 | 0.4945 |

## Ensembling

Two ensembling approaches were built and explicitly compared. If one thought it's Portugal by very little and the other thought it's Spain by a country mile, those are not the same" — i.e., RF's vote-fraction-based probabilities and LR's softmax-based probabilities sit on genuinely different confidence scales, so a flat average silently favors whichever model's outputs happen to run larger in magnitude, independent of which model is actually more trustworthy on a given example.

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

## Latitude / Longitude Regression — Results

### Baseline models

| Model | Mean km error | Median km error |
|---|---|---|
| Linear Regression | 5,507.2 | 4,120.6 |
| Ridge | 5,507.2 | 4,120.5 |
| Decision Tree | 6,165.6 | 5,853.6 |
| Random Forest (untuned) | 5,119.3 | 3,800.5 |

Two things stand out immediately. **Ridge and plain Linear Regression produced virtually identical results** — mean km error matches to one decimal place, median differs by 0.1 km. This is a direct, concrete payoff from the correlation-based feature dropping done earlier in the project: Ridge's whole purpose is to stabilize a model against the kind of multicollinearity that caused concern back when adjacent HSV/LBP bins were found correlated above 0.8. Since those redundant columns were already removed before this stage, Ridge's regularization had very little left to do — confirming that earlier cleanup step actually worked, rather than Ridge simply being redundant in general.

**Decision Tree was the weakest model here**, exactly as it was for both country and continent prediction earlier in this project — a single unconstrained tree consistently underperforms every other model tried across all three targets in this project, a pattern worth noting as a throughline rather than three separate coincidences.

### Tuning

- **Random Forest:** `{n_estimators=200, min_samples_leaf=4, max_features='sqrt', max_depth=25}` — CV score −5,178.4 km (mean).
- **Ridge:** `{alpha=0.1}` — CV score −5,523.7 km (mean).

### Final test set evaluation

| Model | Mean km | Median km |
|---|---|---|
| Random Forest (tuned) | 5,132.0 | 3,773.4 |
| Ridge (tuned) | 5,528.3 | 4,126.1 |

### Ensembling — a genuinely different result than country/continent

| Ensemble | Mean km | Median km |
|---|---|---|
| Simple average (RF + Ridge) | 5,303.4 | 3,895.2 |
| Stacked (meta-LR) | **4,827.7** | **3,612.6** |

This reverses what happened during classification. **Simple averaging made things worse than RF alone** (5,303.4 vs. RF's own 5,132.0 mean km) — unlike averaging two classifiers' probabilities, averaging two regressors' raw coordinate predictions is a literal arithmetic midpoint. When one model (Ridge) is meaningfully worse than the other (RF), that midpoint gets dragged toward the worse prediction rather than benefiting from any "wisdom of crowds" effect. **Stacking, by contrast, produced the best result of the entire regression phase** — the meta-model apparently learned to weight RF's predictions much more heavily than Ridge's, rather than assuming the naive 50/50 trust a flat average forces.

### Feature importance

The top features recurring here — `lbp_pca_2`, `ground_hue_mean`, `hsv_bin_32`, `hsv_bin_1`, `lbp_pca_5`, `ground_grey_frac` — are, with minor reordering, the same features that led both the country and continent importance rankings earlier in this project. This is a meaningful cross-check: it suggests these specific engineered features carry genuinely general geographic signal rather than being useful only for one particular framing of the target. The distribution is flat again, consistent with every prior importance check in this project — no single feature dominates.

### Hedge vs. land-only

| Approach | Mean km | Median km |
|---|---|---|
| Hedge (unconstrained) | 5,132.0 | 3,773.4 |
| Land-only (snapped) | 5,131.3 | 3,777.2 |

The difference is negligible — under 1 km on the mean, and the median is actually marginally *worse* after snapping. A plausible explanation: Random Forest predictions are, by construction, averages of real training-point coordinates within a leaf — meaning RF's raw output is already implicitly biased toward real, populated locations most of the time, unlike a linear model that could more easily extrapolate into open ocean. If that's the mechanism, it would explain why forcing land-only had almost nothing left to fix specifically for RF; the experiment might show a larger effect if repeated on Ridge or LR's raw predictions instead, which extrapolate more freely.

### Mean vs. median — a visible, not just theoretical, pattern

Every model here shows a mean 1,300–2,500 km higher than its own median (e.g., Ridge: 5,528.3 mean vs. 4,126.1 median). This is a direct, visible confirmation of the earlier discussion about why both statistics were worth reporting: a handful of catastrophically wrong guesses (opposite hemisphere, wrong continent entirely) are dragging every mean upward, while the median shows what a "typical" miss actually looks like.

---

## Full-Project Summary and Comparison to Published Benchmarks

### This project's results, all three target levels

| Target | Metric | Best result | Model |
|---|---|---|---|
| Country (111 classes) | Top-1 / Top-2 / Top-3 accuracy | 34.6% / 45.4% / 51.5% | Calibrated ensemble (RF + LR) |
| Continent (6 classes) | Top-1 / Top-2 / Top-3 accuracy | 62.7% / 82.0% / 90.5% | Stacked ensemble (RF + LR) |
| Lat/Long | Mean / Median km error | 4,827.7 / 3,612.6 km | Stacked ensemble (RF + Ridge) |

### External benchmarks — sourced numbers

All five systems below use the same standard evaluation protocol from the geolocation literature: percentage of test images correctly localized within five fixed distance radii (1 km street, 25 km city, 200 km region, 750 km country, 2,500 km continent). This protocol traces directly back to the 2008 IM2GPS paper this project has referenced throughout.

| System | Year | Approach | 1 km | 25 km | 200 km | 750 km | 2,500 km |
|---|---|---|---|---|---|---|---|
| **IM2GPS** (Hays & Efros) | 2008 | Nearest-neighbor retrieval over hand-crafted features — **no neural network** | — | 12.0% | 15.0% | 23.0% | 47.0% |
| **IM2GPS, large-scale revisit** (Hays & Efros) | 2015 | Improved lazy-learning retrieval, still hand-crafted features — **no neural network** | 2.5% | 21.9% | 32.1% | 35.4% | 51.9% |
| **PlaNet** (Weyand, Kostrikov, Philbin) | 2016 | First CNN-based approach; classification over ~26,000 geographic cells | 8.4% | 24.5% | 37.6% | 53.6% | 71.3% |
| **ISNs** (Müller-Budack et al.) | 2018 | Deep CNN with hierarchical scene-context classification | 10.5% | 28.0% | 36.6% | 49.7% | 66.0% |
| **TransLocator** (Pramanick et al.) | 2022 | Transformer-based, combines image + generated semantic segmentation | 11.8% | 31.1% | 46.7% | 58.9% | 80.1% |

*Sources: original IM2GPS and IM2GPS-2015 and PlaNet figures reproduced from Table 1 of Vo, Jacobs & Hays, "Revisiting IM2GPS in the Deep Learning Era," ICCV 2017 (arxiv.org/pdf/1705.04838), evaluated on the original 237-image IM2GPS test set. ISNs and TransLocator figures reproduced from Table 4 of "Visual and Object Geo-localization: A Comprehensive Survey" (arxiv.org/pdf/2112.15202), evaluated on the larger 3,000-image IM2GPS3k test set — note this is a different (larger, harder) test set than the row above it, so treat ISNs/TransLocator's numbers as internally comparable to each other, not strictly apples-to-apples against the 237-image IM2GPS rows.*

**A sixth, especially relevant comparison — PIGEON** (Haas, Skreta, Alberti & Finn, Stanford, CVPR 2024): a CLIP-based model evaluated on a held-out set of **5,000 GeoGuessr-style Street View images** — the closest data source to this project's own dataset of anything found. PIGEON achieved a **median error of 44.35 km**, guessed the correct **country ~92% of the time**, and placed **~40% of guesses within 25 km**. It went on to defeat a professional GeoGuessr player across six consecutive matches. *(Source: Haas et al., "PIGEON: Predicting Image Geolocations," CVPR 2024, arxiv.org/abs/2307.05845; figures corroborated by contemporary reporting in The Register, July 2023.)*

One more directly comparable data point: in PlaNet's own original human-evaluation experiment, it won 28 of 50 rounds against 10 expert GeoGuessr players, with a **median error of 1,100–2,300 km** in that specific match format (via Weyand et al. 2016, as cited in Anderson et al., "Supporting Image Geolocation with Diagramming and Crowdsourcing," par.nsf.gov/servlets/purl/10053400).

### Comparing this project's numbers to the above — with the necessary caveats stated first

Two methodological differences matter before reading anything into these comparisons:
1. **"% within 750 km" is a distance-radius proxy for "correct country," not a literal country match** — it works reasonably well for small-to-medium countries but is a poor proxy for Russia, Canada, the US, Brazil, China, or Australia, all of which exceed 750 km across. This project's country accuracy (34.6%) is a *literal* label match, not a distance proxy — a stricter, more direct definition than most of the table above uses. PIGEON's 92% figure is the one directly comparable exception, since it's also reported as literal country-guess accuracy.
2. **Training data scale differs by orders of magnitude.** PlaNet trained on 91 million images; PIGEON's underlying CLIP model was pretrained on hundreds of millions of image-text pairs before fine-tuning. This project trained on ~86,000 images with classical models on hand-engineered features, run on consumer CPU hardware. The gap below should be read in that light — it reflects a difference in the underlying tools and data scale as much as it reflects room for feature-engineering improvement within the current approach.

With that stated:

- **Country level:** this project's 34.6% literal accuracy sits almost exactly between the original 2008 IM2GPS (23.0% via the 750 km proxy) and its own 2015 improved revision (35.4%) — a genuinely notable result, since it means a classical, hand-engineered-feature pipeline built as this project's explicit homage to that 2008 paper lands in the same performance neighborhood as that paper's own lineage, using a stricter metric. It remains well below PlaNet (53.6%), TransLocator (58.9%), and dramatically below PIGEON (92%).
- **Continent level:** 62.7% sits close to ISNs (66.0%, a genuine 2018 deep-learning system) and above both IM2GPS variants (47.0%, 51.9%), while remaining below PlaNet (71.3%) and TransLocator (80.1%). This is this project's strongest relative showing against published deep-learning benchmarks.
- **Distance error:** this is the weakest showing by comparison. This project's best median error (3,612.6 km) is roughly **1.6× to 3.3× worse** than PlaNet's median error against human experts (1,100–2,300 km) — itself a 2016 system already considered outperformed today — and roughly **82× worse** than PIGEON's 44.35 km median. Every model built in this phase, even the best one, produces a typical miss larger than the 2,500 km continent threshold every paper above uses as its *most forgiving* bucket — meaning that on a typical guess, this project's regression models are, on average, still further off than "continent" by the field's own definition of that word.

The country and continent classification results hold up as a legitimate, if modest, echo of the project's founding inspiration — landing in the same range as the very paper it set out to rebuild a scaled-down version of. The direct coordinate-regression task is where the gap to modern systems is most stark, consistent with the caveat raised earlier in this project that direct lat/lon regression is not what the field's strongest systems actually do, and that the field's own trajectory (IM2GPS → PlaNet → PIGEON) moved specifically *away* from raw coordinate prediction and toward classification-based framing for exactly this reason.

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
