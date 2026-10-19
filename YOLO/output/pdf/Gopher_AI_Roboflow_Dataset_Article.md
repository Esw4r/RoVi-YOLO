# Gopher-AI: A five-class road damage dataset with self-collected images on Roboflow

Dataset article | Revised 15 October 2026

## Abstract
Road damage detection depends on image collections that describe the appearance and location of pavement defects under relevant capture conditions. This article presents a descriptive study of the pothole (Gopher_AI) project published by nothighs Workspace on Roboflow Universe. The public project lists 460 images, an object-detection task and five classes: Pothole, alligator crack, longitudinal crack, other corruption and transverse crack. Selected public samples show both daylight and nighttime road scenes from a vehicle-facing perspective. The image browser displays annotation overlays and Train, Valid and Test assignments. The dataset owner reports that the collection includes some self-collected images. The public overview lists CC BY 4.0, while no generated dataset version or project-trained model is displayed. The article describes the dataset, its visible content and a proposed preparation and evaluation workflow. It also identifies the metadata needed to establish image provenance, class balance and independent test coverage. The collection offers a basis for studying multi-class road damage detection; model accuracy and the benefit of self-collected images remain to be measured on a documented dataset version.

Keywords: Road damage; Potholes; Roboflow Universe; Object detection; Self-collected images; Dataset characterization


## 1. Background

Automated road inspection requires more than identifying whether an image contains damaged pavement. A useful detection system should locate individual defects and distinguish the types of damage visible in the scene. The image collection and its annotation vocabulary therefore define what the system can learn and what an evaluation can reasonably claim.

The pothole (Gopher_AI) project provides a public collection for this task on Roboflow Universe [1]. Although its title emphasizes potholes, the project includes five road-damage labels. This broader vocabulary makes it relevant to a multi-class inspection workflow in which cracks and other surface defects are considered alongside potholes.

The dataset owner reports that some images are self-collected. Such material can support a study of the roads and conditions encountered by the project, provided that its origin is documented and related images are kept together during dataset partitioning. Its contribution should ultimately be assessed through a controlled comparison rather than inferred from image count alone.

The objective of this article is to characterize the linked Roboflow project using its public metadata and visible sample content. The article follows a dataset-study design: it reports the resource that is available, explains its potential uses and describes a proposed evaluation workflow. It does not report a completed model-training experiment.


## 2. Materials and methods


### 2.1. Source and scope of inspection

The project overview and public image browser were inspected on 15 October 2026 [1,2]. The review recorded the project title, workspace attribution, task, image total, class names, displayed license, version status and model status. Publicly visible filenames and partition labels were reviewed, and three sample thumbnails linked from the overview were retrieved for qualitative illustration.

Project metadata were distinguished from visual observations and owner-provided information. The presence of self-collected data is based on the owner's statement; the public page does not identify that subset. The image examples are a small convenience sample and were not used to estimate the prevalence of lighting conditions or damage categories across the entire collection.


Table 1. Dataset profile reported by the public project overview [1].
Attribute | Recorded information
Project title | pothole (Gopher_AI)
Dataset publisher | nothighs Workspace
Hosting platform | Roboflow Universe
Task | Object detection
Image total | 460
Class total | 5
Displayed license | CC BY 4.0
Dataset versions | 0 displayed
Project models | 0 displayed


### 2.2. Damage categories

The five class names shown by the project are reproduced in Table 2. The explanatory descriptions are working interpretations for this article, rather than a published annotation handbook. The original class strings should be preserved when inspecting an export, and the numerical class order should be read from that export rather than assumed from the order on the webpage.


Table 2. Published class names [1] and their working interpretation.
Class name | Interpretation for review
Pothole | A localized cavity or depression in the pavement.
alligator crack | A connected network of cracks forming a patterned damaged region.
longitudinal crack | An elongated crack running approximately along the road direction.
other corruption | An additional project category whose inclusion criteria need definition.
transverse crack | An elongated crack running approximately across the road direction.

The distinction between longitudinal and transverse cracking depends on road direction, which may appear oblique in a camera view. Alligator cracking may occupy a broader region than a single narrow crack. Annotation review should therefore address orientation, the grouping of connected defects and the boundary of each labelled region. These are proposed review considerations; the page does not state how its contributors resolved them.


### 2.3. Visible image content

The selected public thumbnails include two daylight road scenes and one nighttime scene (Fig. 1). The daylight examples contain a forward view of the road, roadside buildings and vegetation, with part of the vehicle visible along the lower edge. The nighttime example has a comparatively dark surrounding scene and an illuminated area of road ahead. These observations describe the reproduced samples only.


Figure 1. Figures are source thumbnails, shown without enhancement or added detections. They illustrate scene appearance, not model predictions or class prevalence. Source: nothighs Workspace, pothole (Gopher_AI), Roboflow Universe [1], listed CC BY 4.0. Panels (a)-(c): image IDs 0OMDvvQj151NoN4EjfAU, 0OxVeGtt3jvnQpFqXEV5 and 0SuNzTjtqnoCGWoSp8so.

In the public browser, the annotation-display option was enabled and rectangular overlays were visible on several preview images. This is consistent with the project's object-detection task. The review did not establish whether every image has complete annotations, how many objects each class contains or whether empty-label images represent undamaged roads or incomplete annotation.


### 2.4. Image organization and provenance

Visible image cards carried Train, Valid and Test labels [2]. Examples included NO20221108-211950-000171F_00001.jpg in Train, the corresponding _00008.jpg file in Valid and _00014.jpg in Test. Their common filename prefix is a reason to check capture relationships, but it does not establish that the images duplicate one another or that they were taken at a particular location.

Full split totals were not established from the accessible public preview. The browser requested sign-in to browse all 460 images. Consequently, no train-validation-test percentages, per-class frequencies or self-collected subset counts are assigned in this article. A filename that resembles a timestamp is not treated as verified acquisition metadata.

For a reproducible study, a source manifest should associate each image with its acquisition session, route group, original source and self-collected status where known. That record would make it possible to measure the contribution of self-collected images and to check whether related frames occur in different partitions. The manifest is a proposed addition, not a resource documented on the inspected page.


## 3. Results: dataset characterization

The public overview identifies a 460-image, five-class object-detection resource [1]. This is the principal quantitative result of the dataset review. Image count and class count describe the scope of the resource; they do not establish the number of damage instances, class balance or the independence of the photographed scenes.

The visible sample content supports the presence of both daylight and nighttime imagery. These examples suggest that lighting variation is a relevant dimension for a future analysis. They do not establish the number of nighttime images, the full range of weather conditions or the geographical coverage of the dataset. Those quantities require a complete inventory with verified metadata.

The overview displayed zero dataset versions and zero models. It also indicated that a project description had not been published. Thus, there was no published version identifier or project-trained evaluation result to associate with the collection. The page's generic detection demonstration is labelled as being powered by a general detection model; it is not evidence of a model trained on these 460 images.


Table 3. What the inspected project supports and what remains to be established.
Established from source | Not established
460 listed images | Total annotated objects and class frequencies
Five named classes | Detailed annotation policy and class-ID order
Day and night examples | Full lighting-condition distribution
Train / Valid / Test labels | Complete split totals and independent capture groups
Owner reports self-collected content | Number and provenance of those images
No published model displayed | Precision, recall, mAP or inference speed on this collection

The current findings support publication of a dataset description and preparation of a future training study. They do not support numerical claims about detection accuracy, improvements from self-collected images or suitability for unattended deployment. Those claims require an experiment linked to an identifiable dataset version.


## 4. Discussion


### 4.1. Value for road damage research

The project brings five damage categories into one named collection and supplies a public location for inspecting its content. This provides a practical starting point for developing an annotation policy and testing a multi-class detector. The difference between its pothole-focused title and its broader label set should be made explicit when defining the research objective.

The daylight and nighttime samples are useful because they motivate evaluation by capture condition. A single overall score could conceal different behavior under bright road illumination and limited nighttime visibility. If sufficient independently collected examples are available, reporting results separately for these conditions would make the evaluation more informative.


### 4.2. Proposed preparation and evaluation workflow

Before training, the collection should be reviewed for duplicate images, inconsistent labels and related frames from the same capture sequence. Annotation review should document how a defect is bounded, how overlapping damage types are handled and what qualifies as other corruption. The number of annotated objects and images containing each class should then be reported separately.

Partitioning should operate on independent capture groups where possible. Images of the same road segment or closely related video frames should be kept in one partition. A versioned export should retain the split manifest and preprocessing settings. Any augmented copies should be generated from the training split after partitioning, so that transformed versions of a test image cannot enter training.

A future model study should select its configuration using training and validation data, then evaluate once on the frozen test set. Suitable outcomes include per-class precision, recall and average precision, together with a clearly defined intersection-over-union matching rule. Reporting missed potholes and false alarms separately would complement an aggregate score. No values for these measures are available from the inspected project.

The role of self-collected data can be tested by comparing training with and without the eligible self-collected training subset while using the same independent test set. The comparison should hold the model, training budget and evaluation procedure constant. If possible, repeated runs and uncertainty estimates should be reported. This design would address the contribution of the added images directly.


### 4.3. Limitations

The review used the public overview, accessible image previews and three illustrative thumbnails. It did not download a complete annotated dataset or verify every image. The samples were not randomly selected and cannot support a distributional claim. The number, location and acquisition conditions of self-collected images were not supplied, and the project's detailed annotation policy was not available.

No detector was trained or evaluated for this article. The observed absence of a public model does not imply that the owner has never trained a model elsewhere; it means that this public resource did not supply a verifiable result. A subsequent experimental article should link every reported metric to the dataset version, checkpoint and evaluation configuration used.


## 5. Conclusion

The pothole (Gopher_AI) project on Roboflow Universe is a 460-image object-detection collection covering potholes, alligator cracks, longitudinal cracks, transverse cracks and other corruption. Public examples include daylight and nighttime road views, and the owner reports a self-collected contribution. The collection provides a basis for a multi-class road damage study. Documented provenance, annotation review, a frozen dataset version and independent evaluation are the next steps needed to establish its effectiveness for model development.


## Data availability and attribution

The dataset overview and image browser are linked in references [1] and [2]. The project credits nothighs Workspace and displays CC BY 4.0. Figure 1 reproduces three public thumbnails at reduced display size, with no enhancement or added annotations. The self-collected contribution is attributed to the owner's statement; individual contributors are not identified by the public project metadata.


## Declarations

Individual author names, affiliations, funding information and conflict-of-interest statements were not supplied. These declarations remain for author completion before submission. The manuscript was prepared with AI assistance and requires author review.


## References

[1] nothighs Workspace. pothole (Gopher_AI) Dataset. Roboflow Universe; October 2026. Accessed 15 October 2026. https://universe.roboflow.com/not00high-gmail-com/pothole-gopher_ai-eh26c

[2] nothighs Workspace. pothole (Gopher_AI): public image browser. Roboflow Universe. Accessed 15 October 2026. https://universe.roboflow.com/not00high-gmail-com/pothole-gopher_ai-eh26c/browse
