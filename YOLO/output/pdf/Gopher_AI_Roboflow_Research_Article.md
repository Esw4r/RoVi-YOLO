# Gopher-AI: Road damage detection with a Roboflow dataset containing self-collected images

Research manuscript draft | 15 October 2026

## Abstract
Road damage detection requires image collections that represent both established benchmarks and the intended deployment environment. This article characterizes the public pothole (Gopher_AI) Roboflow project and reports a separate, previously saved evaluation of the existing Gopher-AI detector. The public project contained 460 images and five damage classes at inspection; its owner reports that some images were self-collected. No published dataset version or project-trained model was listed. Local evaluation records describe a YOLO11n detector tested on 5,076 images from a processed RDD-based collection, with 8,325 annotated objects. The recorded mean average precision was 49.04% at an intersection-over-union threshold of 0.50 and 24.50% across thresholds of 0.50-0.95. Pothole average precision at 0.50 was 35.82%. These values establish an existing baseline and do not measure performance on the 460-image Roboflow collection. A provenance-aware preparation and evaluation protocol is proposed to test the contribution of self-collected images. The available evidence supports dataset development and further validation; it does not yet establish improved accuracy from local data or readiness for unattended road inspection.

Keywords: Road damage; Pothole detection; YOLO11n; Roboflow; Self-collected images; Dataset provenance


## 1. Background

Image-based road inspection can support the identification of damaged pavement by locating defects within ordinary road photographs. The task differs from deciding whether an entire image is damaged: an object detector must assign a damage category and estimate the position of each visible instance. Missed defects and incorrect localizations therefore require separate attention when interpreting performance.

The RDD2022 publication describes a multinational resource containing 47,420 images from six countries and annotations for four road-damage categories [2]. Such benchmarks provide a useful starting point, but an evaluation on one collection does not establish performance on a newly assembled local collection. Camera viewpoint, pavement texture, lighting and the way defects are annotated may differ between sources.

The pothole (Gopher_AI) project on Roboflow provides a five-class collection for this application [1]. The owner reports that it includes some self-collected images. This contribution is potentially useful for studying the conditions encountered by the project, provided that the origin of each image and its relationship to the evaluation split are recorded. The number and acquisition conditions of these self-collected images were not supplied.

This article has two objectives: to document the publicly verifiable state of the Roboflow collection and to interpret the existing Gopher-AI evaluation as a baseline for subsequent work. A further objective is to specify how the contribution of self-collected data could be tested. The study does not assume that the Roboflow images were used to train the saved detector, and it reports no new training experiment.


## 2. Materials and methods


### 2.1. Study design and evidence sources

A descriptive dataset review was combined with a secondary analysis of saved project evaluation records. The public Roboflow overview and image browser were inspected on 15 October 2026. The local evidence consisted of the dataset audit, model metadata, full-test validation summary, fixed-threshold analysis and the evaluation script in the Gopher-AI workspace [4]. Values were transcribed from these records; the detector was not retrained or re-evaluated for this article.

The evidence was divided into three categories. Publicly displayed project metadata support the description of the Roboflow collection. The presence of self-collected material is an owner-provided statement. Numerical detection results originate only from the saved local evaluation. Keeping these categories separate prevents a dataset description from being mistaken for a completed experiment.


Table 1. Publicly observed Roboflow project characteristics [1].
Characteristic | Observation
Project | pothole (Gopher_AI)
Workspace | nothighs Workspace
Task | Object detection
Images | 460
Classes | 5
Published versions | 0 displayed
Project models | 0 displayed
Listed license | CC BY 4.0
Self-collected subset | Present according to owner; count not supplied
Full split totals | Not established from the public view


### 2.2. Dataset description and annotation vocabulary

The public project lists Pothole, alligator crack, longitudinal crack, other corruption and transverse crack. These names correspond to the five labels recorded in the existing detector metadata. Name-level compatibility is useful, but it does not establish that class identifiers, annotation policies or image sources are identical. An exported dataset should therefore be checked before labels are used for training or evaluation.

For this project, potholes are interpreted as localized pavement depressions or cavities; longitudinal and transverse cracks describe elongated fractures with different orientations relative to the road; alligator cracks describe interconnected cracking. These are working interpretations for annotation review. The public page did not provide a detailed annotation handbook, and the scope of other corruption requires an explicit operational definition before combining sources.

The public image browser showed Train, Valid and Test assignments among the visible images. Some visible filenames shared a common sequence-like prefix across different assignments. That observation motivates a review for related frames, but filenames alone do not prove duplicated content or evaluation leakage. The accessible preview did not establish complete split counts or the independence of its image groups.

The project overview displayed no published dataset version and no project-trained model. Consequently, an immutable export identifier, preprocessing recipe, augmentation history and collection-specific benchmark result could not be cited. The generic detection demonstration on the public page was not treated as evidence that a model had been trained on this collection.


### 2.3. Preparation of self-collected images

The proposed preparation protocol begins with a per-image provenance manifest. Each record should identify the source collection, original filename, collection session, road segment or route group, capture device when known, and whether the image is self-collected or obtained from another source. Missing acquisition metadata should remain explicitly unknown. The public image total must not be presented as the number of self-collected images.

Annotation review should define what constitutes a distinct defect, how partially visible damage is bounded and when uncertain cases are excluded. Reviewers should check that bounding boxes are inside the image, that class identifiers follow the intended mapping and that duplicate annotations are removed. Images containing no target damage should remain distinguishable from images that have not yet been annotated.

Before partitioning, exact duplicates should be identified by image hashes, while near-duplicate frames and repeated views should be reviewed using acquisition groups and visual similarity. All images derived from the same original photograph or closely related capture sequence should remain in one partition. Augmented copies should be generated only from the training partition after this grouping step.

The exact train-validation-test allocation should be chosen after the number of independent groups and per-class examples is known. Assigning a percentage of 460 images without this information could create a misleading impression of statistical independence. A frozen test set should contain independently collected scenes that are not used to select augmentation, confidence thresholds or model checkpoints.


### 2.4. Existing local detector and dataset

The local project uses the nano object-detection variant of Ultralytics YOLO11 [3]. Its saved metadata report 2,590,815 parameters and five output classes [4]. The project README describes an RDD-based preparation process that removes selected non-dashcam sources, limits background-only training images and resizes large images. The audited collection is therefore a processed project dataset, not the complete original RDD2022 release.

This distinction also applies to taxonomy. The original RDD2022 description identifies four damage types [2], whereas the local export and Roboflow project both contain five labels, including other corruption. The additional category is treated here as a project-export label. Its provenance and correspondence to upstream annotations should be documented before comparing results with published four-class benchmarks.


Table 2. Saved local dataset audit; these counts do not describe the 460-image Roboflow project [4].
Partition | Images | Objects | Backgrounds
Train | 16,726 | 40,197 | 1,000
Validation | 5,113 | 8,422 | 1,837
Test | 5,076 | 8,325 | 1,790

The saved audit reports no shared filenames or exact image-content hashes between its train, validation and test partitions. It also records one excess duplicate image within training, one invalid training annotation row, duplicate annotations in two training images and a duplicate annotation in one validation image. No test annotation issues were reported by that audit. These are properties of the recorded local snapshot; they do not constitute an audit of the Roboflow collection.


### 2.5. Recorded training and validation configuration

Checkpoint metadata specify a training configuration of up to 100 epochs, early-stopping patience of 10, batch size 16 and input image size 512 pixels. The optimizer was AdamW with an initial learning rate of 0.0015; automatic mixed precision was disabled. The recorded checkpoint epoch was 78 using zero-based numbering, corresponding to the 79th epoch. This identifies the saved checkpoint state rather than proving that every planned epoch was completed.

The saved configuration includes horizontal flipping with probability 0.5, no vertical flipping and zero rotation. Mosaic was configured to close during the final 10 epochs. These values describe the existing run; their effectiveness on the Roboflow collection remains untested. A subsequent adaptation study should retain an unchanged baseline configuration unless a change is explicitly included in the experimental comparison.

The local validation script specifies the test partition, image size 512, batch size 8, full-precision inference, a confidence floor of 0.001, non-maximum suppression IoU of 0.70 and at most 300 detections per image. The environment record identifies an NVIDIA GeForce GTX 1650 with approximately 4 GB of memory, Python 3.12.10, PyTorch 2.11.0+cu128 and Ultralytics 8.4.90 [4].


### 2.6. Outcome measures and comparison protocol

Detection precision is TP/(TP + FP), recall is TP/(TP + FN), and F1 is 2PR/(P + R), where TP, FP and FN denote true positives, false positives and false negatives. Intersection over union (IoU) measures the overlap between predicted and reference boxes. Average precision summarizes the precision-recall curve for one class; mean average precision averages that quantity across classes.

The saved validation output provides mAP at IoU 0.50 and mAP averaged over IoU thresholds from 0.50 to 0.95. A separate analysis provides counts at a confidence threshold of 0.25 and a matching IoU of 0.50. These fixed-threshold counts must not be interchanged with the validator's confidence-swept summary. The article reports both only with their respective definitions.

To test the contribution of self-collected images, a future experiment should compare a baseline trained without those images with a model trained using eligible self-collected training images. Both models should be evaluated on the same frozen, independent test set. The class mapping, test images, preprocessing and metric definitions should remain fixed. Results should be separated by source and damage category, with uncertainty estimated over independent acquisition groups when sufficient groups exist.


## 3. Results


### 3.1. Verified dataset status

The Roboflow overview establishes the existence of a 460-image, five-class project and supplies a public dataset citation [1]. It does not establish the number of annotated objects per class, the share of self-collected material or the complete split totals. No trained-project performance was available in the inspected public metadata. The dataset findings are therefore descriptive rather than an accuracy assessment.

The separate local audit contains 26,915 images across its three partitions and 56,944 annotated objects. The test partition contains 5,076 images and 8,325 objects. Of those images, 1,790 were recorded as backgrounds, approximately 35.3% of the test set. Including these scenes makes false detections on images without target damage relevant to practical interpretation.


### 3.2. Existing baseline detection performance

For the saved five-class local evaluation, precision was 54.81%, recall was 49.07%, mAP@0.50 was 49.04% and mAP@0.50:0.95 was 24.50%. The precision and recall values in this paragraph are the validator's summary outputs. They do not represent the independently calculated fixed-confidence results reported below, and none of these values was measured on the Roboflow collection.


Table 3. Per-class average precision on the local test set [4]. AP values are percentages.
Class | Objects | AP50 | AP50-95
Longitudinal crack | 3,293 | 43.15 | 22.06
Transverse crack | 1,311 | 39.59 | 16.57
Alligator crack | 1,397 | 59.50 | 29.86
Other corruption | 1,419 | 67.16 | 39.34
Pothole | 905 | 35.82 | 14.68

Other corruption had the highest recorded AP50, while pothole had the lowest. The difference between the overall AP50 and the stricter IoU-averaged result was 24.54 percentage points. The decline indicates sensitivity to stricter localization requirements; the summaries alone cannot identify whether box extent, small-object visibility or annotation differences explain the gap.

At confidence 0.25 and matching IoU 0.50, the separate analysis recorded 3,988 true positives, 3,082 false positives and 4,337 false negatives. The resulting micro-averaged precision, recall and F1 were 56.41%, 47.90% and 51.81%, respectively. Thus, more annotated objects were missed than matched at that operating point.

For potholes at the same fixed threshold, 298 objects were matched and 607 were missed, with 255 false positives. Precision was 53.89%, recall 32.93% and F1 40.88%. These results identify pothole recall as a specific area for further development, but they do not show that collecting more images alone will resolve the limitation.


### 3.3. Recorded processing time

The saved validator reports approximately 0.43 ms per image for preprocessing, 6.92 ms for inference and 1.54 ms for postprocessing. The complete validation call took 207.94 seconds for 5,076 images. Its wall-clock throughput was therefore approximately 24.4 images per second under that evaluation setup. These measurements differ in what they include and should be reported separately.

The inference-stage time is a batched validation measurement. It is not an end-to-end camera latency measurement and does not include a deployed application's complete acquisition, decoding, communication and display path. No embedded-device benchmark or live road-inspection latency was measured for this article.


## 4. Discussion


### 4.1. Contribution of the Roboflow collection

The principal contribution of the Roboflow project at its current documented stage is an identifiable collection with a compatible set of damage labels and owner-reported self-collected content. It provides a basis for organizing project-specific data and for designing an adaptation experiment. The available evidence does not establish an accuracy improvement, a novel detector architecture or a performance advantage over competing systems.

Self-collected images can make an experiment more relevant when they represent the intended roads and capture conditions. However, collecting many nearby frames may increase image count without adding much independent information. Reporting the number of routes, capture sessions and distinct road segments would therefore provide useful context alongside the number of photographs.

A meaningful next experiment would ask whether adding the eligible self-collected training subset improves detection on independent local scenes. A second question is whether such adaptation preserves performance on the original benchmark-derived test set. Improvement on one source should not be assumed to imply improvement on all sources. Source-specific reporting would make this tradeoff visible.


### 4.2. Interpretation of the existing baseline

The existing results show that class-level reporting is necessary. An aggregate mAP of 49.04% at IoU 0.50 conceals lower performance for potholes and transverse cracks than for other corruption and alligator cracks. Because the proposed project emphasizes potholes, a conclusion based only on the five-class mean would omit a central weakness of the present detector.

The fixed-threshold results also show the practical importance of missed detections. At the reported operating point, pothole recall was approximately one third. A lower confidence threshold may recover some missed instances while increasing false alarms, but the appropriate operating point must be selected using validation data and the intended review workflow. The final test set should not be used to optimize that threshold.

Reviewing false positives and false negatives should precede claims about their causes. Shadows, repairs, water and small distant defects are plausible inspection categories, but this article did not conduct a systematic visual error taxonomy. A future review could sample errors by class, size and acquisition condition, record the decisions of reviewers and distinguish annotation omissions from detector failures.


### 4.3. Data independence and reproducibility

Exact-hash separation in the local audit is useful evidence against identical-file overlap. It does not rule out near-duplicate scenes, consecutive video frames or repeated road segments across partitions. Likewise, different Train, Valid and Test labels on the public Roboflow browser do not by themselves establish independent sampling. An acquisition-group manifest is needed to address those concerns directly.

The absence of a published Roboflow version is also consequential for reproducibility. A changing project page cannot uniquely identify the images and annotations used in an experiment. Before training, the collection should be frozen into a versioned export with a retained class mapping, split manifest, source counts and preprocessing configuration. The trained checkpoint and evaluation outputs should be linked to that export.

The fifth class deserves particular attention when combining data. If other corruption collects heterogeneous defects, its annotation boundaries may differ across contributors. An explicit inclusion and exclusion policy, supported by reviewed examples, would help prevent a model from learning inconsistent targets. Class compatibility should be confirmed semantically as well as by matching names.


### 4.4. Limitations and intended use

This article is limited by its reliance on public project metadata and pre-existing local evaluation records. The complete Roboflow collection was not downloaded or independently audited, and its provenance, full split sizes and per-class object counts remain unverified. The owner's self-collected contribution is acknowledged without assigning an unsupported size, location, device or collection period.

The baseline represents one saved detector and one processed local dataset. No new random-seed repetitions, confidence intervals, architecture comparisons or self-collected-data ablations were performed. The local model's training relationship to the Roboflow project was not established. Consequently, the baseline values cannot be attributed to the 460-image collection or used to claim that self-collected images improved detection.

The intended near-term use is assisted inspection: detections can be presented to a person for review, while missed and uncertain cases inform further data collection. Bounding boxes alone do not measure pothole depth, repair cost or physical severity. Those outputs would require additional labels, measurements and validation beyond the evidence considered here.


## 5. Conclusion

The pothole (Gopher_AI) Roboflow project contains 460 publicly listed images and five damage classes, with some self-collected material reported by the owner. Its current public metadata support a dataset description but not a trained-model performance claim. The existing local Gopher-AI evaluation provides a separate baseline: mAP@0.50 of 49.04%, mAP@0.50:0.95 of 24.50% and pothole AP50 of 35.82%.

The next research step is to document provenance, review annotation consistency, freeze an independently partitioned dataset version and compare training with and without the self-collected subset. That experiment would establish whether the additional images improve performance in the intended environment. Until then, the article supports further development and transparent evaluation rather than a conclusion of demonstrated local-data benefit.


## Data and code availability

The public Roboflow project is available at the link in reference [1]; its overview lists CC BY 4.0. The local Gopher-AI evaluation artifacts used here are identified in reference [4]. These local records were available for manuscript preparation but their public availability is not established. A versioned Roboflow export and the corresponding trained checkpoint were not available for this analysis.


## Funding and disclosure of interest

Funding information and conflict-of-interest declarations were not supplied. These statements require confirmation by the authors before submission.


## Authors' contributions

The dataset owner reports contributing self-collected images. Individual authorship, affiliations and contribution roles were not supplied and are therefore not assigned in this draft. The manuscript was prepared with AI assistance and requires author review of data interpretation, citations and declarations.


## References

[1] nothighs Workspace. pothole (Gopher_AI) Dataset. Roboflow Universe; 2026. Public overview and image browser, accessed 15 October 2026. https://universe.roboflow.com/not00high-gmail-com/pothole-gopher_ai-eh26c

[2] Arya D, Maeda H, Ghosh SK, Toshniwal D, Sekimoto Y. RDD2022: A multi-national image dataset for automatic Road Damage Detection. arXiv:2209.08538; 2022. https://doi.org/10.48550/arXiv.2209.08538

[3] Jocher G, Qiu J. Ultralytics YOLO11. Software; 2024. Official model documentation, accessed 15 October 2026. https://docs.ultralytics.com/models/yolo11/

[4] Gopher-AI project workspace. Saved local evaluation artifacts: output/evaluation/audit.json; model_metadata.json; baseline_metrics.json; analysis.json; and evaluation/run_baseline.py. Local project records inspected 15 October 2026. The validation script identifies gopher-ai_v1.pt as the evaluated checkpoint.
