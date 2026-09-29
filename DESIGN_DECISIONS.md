# Design decisions (and the questions they answer)

**Why region-grouped cross-validation, not a random split?**
Test regions are separate places. Tiles sit on a 32-px grid with 64-px windows, so neighbours share
50% of their pixels (verified: every overlapping pair matches exactly). A random split puts
near-duplicates on both sides and hugely overstates accuracy.

**Why gradient-boosted trees rather than a CNN?**
Compute and evidence. With hand-built spectral and texture features, LightGBM reached 0.97+ on held-out
regions. A small CNN was prototyped, but on the available hardware it could not be validated with
the same 5-fold region protocol, so it is not part of the submission. Unvalidated models were not used.

**Why texture features?**
Error analysis showed region R03's Cropland was mostly predicted as Grassland (bare or harvested fields
on the image date have grass-like colour). Fields have straight, aligned edges and regular parcels.
Orientation coherence, FFT anisotropy, GLCM and LBP capture that. Cropland F1 went from 0.67 to 0.97.

**How is class imbalance handled?**
Inverse-frequency class weights in training; every result is reported per class; the post-processing
region prior has a floor so rare classes are not suppressed.

**Why neighbour smoothing?**
Tile IDs encode position in a region grid, and land cover is spatially coherent. Averaging a tile's
probabilities with its neighbours' removes isolated errors (+0.05 on the 5-class score).

**What is the "region prior" (EM)?**
Class proportions vary hugely between regions. EM re-estimates each region's class mix from the model's
own predictions and re-weights them (Saerens et al., 2002). Its strength was tuned by CV; the final
tuned setting turns it off in favour of smoothing alone.

**Why is region R06 excluded from the 5-class score?**
R06 holds 98% of all Shrubland. When R06 is held out, the model has almost no Shrubland to learn from,
so its 4,090 tiles become false Grassland, which destroys Grassland precision. In the real test R06 is
in training, so that failure cannot occur. Both scores are always reported.

**How was Shrubland handled in the final submission?**
Training Shrubland is 98% one desert region, so the main model calls other scrub "Grassland".
A probe showed this directly (NR04: +0.126 public). Several candidate rules were tested on held-out
training regions first: terrain ruggedness failed (reversed between regions); a Shrubland-vs-Grassland
classifier learned lighting, not vegetation. What worked: Shrubland is woody, so among Grassland
predictions the model's runner-up is Forest for scrub and Cropland for grass fields. With random-convolution
texture features this separates the classes in every held-out region (AUC 0.72-1.00) and matches both
public cases. It is applied as one rule to every tile; no region is edited by hand. The confidence bar
is 0.9: at 0.5 it also fired on known Grassland in R14/R15 (public 0.85440 -> 0.84746).

**Isn't using the public leaderboard overfitting?**
Public and private regions are different places, so tuning to public tiles alone cannot help the
private score. Probes were used to test hypotheses about what the model gets wrong in new regions
(probes each changing one region or one rule), and only a general lesson was carried to the private regions,
as an explicit, documented hedge.

**Is self-training allowed / is it leakage?**
It uses only the supplied, unlabelled test images, never labels. In validation it was simulated
honestly: pseudo-labels for a held-out fold came only from models that never saw that fold.

**What would you do with more time?**
Validate a CNN on 128x128 context windows cut from the stitched region mosaics; multi-seed and
multi-model ensembles; better Shrubland/Grassland separation, which likely needs multi-temporal data.
