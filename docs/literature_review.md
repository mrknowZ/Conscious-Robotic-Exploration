# Literature Review — Track 1: Aerial Vision & Cost-Map

## Paper 1: UAVid: A Semantic Segmentation Dataset for UAV Imagery
Lyu et al., 2020 — https://arxiv.org/abs/1810.10438

**Problem it addresses:** Most semantic segmentation datasets at the time were built either
from ground-level driving scenes (Cityscapes, CamVid) or from nadir (straight-down) satellite/
airborne imagery. Neither setup matches what a UAV actually sees, since drones typically fly
at oblique angles and moderate altitude, capturing both the top and side of objects at once.
This creates a real gap for anyone trying to train a segmentation model specifically for drone
footage.

**Method summary:** The authors collected 30 video sequences from an oblique UAV perspective
at roughly 50m altitude over urban environments, densely labeling 8 classes (building, road,
static car, moving car, tree, low vegetation, human, background clutter). They benchmarked
several segmentation architectures on this data, with a multi-scale dilation-based network
performing best, highlighting that large scale variation and temporal consistency (since frames
come from video) are the dataset's core challenges.

**Relevance to our cost-map pipeline:** This is the closest match to our actual camera setup —
oblique UAV views rather than satellite nadir shots — so it's a strong candidate for either
direct fine-tuning or as a sanity check against domain shift once we compare against our own
real-world drone footage. The dataset's emphasis on scale variation is also directly relevant
since our cost-map needs to handle both near-field and far-field terrain features consistently.

## Paper 2: Informative Path Planning for Active Learning in Aerial Semantic Mapping
Rückin et al. — https://arxiv.org/pdf/2203.01652

**Problem it addresses:** Supervised segmentation models need large amounts of labeled data,
which is expensive and slow to collect for aerial/terrain imagery. The paper tackles this by
having the UAV itself decide which images are most worth labeling, rather than labeling
everything collected on a fixed flight path.

**Method summary:** The authors use a Bayesian approach to estimate the model's own uncertainty
during flight, feed that uncertainty into a live terrain map, and then use it as the planning
objective for the drone's next moves — actively steering the UAV toward the areas where the
model is least confident. This reduces both the labeling burden and improves final model
performance versus a static, pre-planned coverage path.

**Relevance to our cost-map pipeline:** This gives us a cheaper way to close the domain gap
discussed earlier — instead of randomly picking images from our own test site to annotate for
validation, we could prioritize the frames where our segmentation model is least confident,
getting more value out of a small manually labeled set. It's also conceptually close to our own
resource-aware exploration goal: using model uncertainty (or terrain cost) to guide where a
robot should go next, rather than covering ground blindly.
