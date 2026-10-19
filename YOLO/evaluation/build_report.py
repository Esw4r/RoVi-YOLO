"""Render the measured evaluation into a framework-traceable PDF report."""
from pathlib import Path
import json, math, html, hashlib, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'output/evaluation'; O=ROOT/'output/pdf'
O.mkdir(parents=True,exist_ok=True)
FIG=E/'report_figures'; FIG.mkdir(exist_ok=True)
def read(name): return json.loads((E/name).read_text())
A=read('audit.json'); B=read('baseline_metrics.json'); D=read('analysis.json'); X=read('extended_results.json'); M=read('model_metadata.json')
NAVY='#183047'; TEAL='#007C83'; ORANGE='#D67832'; MUTED='#536779'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.titleweight':'bold'})
def pct(v): return f'{v*100:.2f}%'
def num(v): return f'{v:,}'
def figsave(name):
    plt.savefig(FIG/name,dpi=180,bbox_inches='tight',facecolor='white'); plt.close()

# Measured, static figures for an exportable report.
names=list(D['subgroups']); values=[D['subgroups'][n]['recall']*100 for n in names]
plt.figure(figsize=(8.2,3)); plt.barh([{'United':'United States','Czech':'Czech Republic'}.get(n,n) for n in names],values,color=TEAL)
plt.xlabel('Detection recall (%) at confidence 0.25 / IoU 0.50'); plt.xlim(0,100)
for i,v in enumerate(values): plt.text(v+1,i,f'{v:.1f}%',va='center')
figsave('subgroups.png')
bins=D['detection_calibration']['bins']; plt.figure(figsize=(7.6,3.3))
plt.plot([0,1],[0,1],color='#AAB8C3',linestyle='--',label='Perfect calibration')
plt.plot([b['confidence'] for b in bins],[b['precision'] for b in bins],'o-',color=TEAL,label='Measured detection precision')
plt.xlabel('Mean detection confidence'); plt.ylabel('Fraction correct'); plt.xlim(.2,1); plt.ylim(0,1); plt.legend(frameon=False,fontsize=9)
figsave('calibration.png')
if 'interpretability' in X:
    from PIL import Image as PILImage
    im=np.asarray(PILImage.open(E/'occlusion_source.jpg')); heat=np.load(E/'occlusion_map.npy')
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    axs[0].imshow(im); box=X['interpretability']['box']
    from matplotlib.patches import Rectangle
    axs[0].add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor='#00FFB2',linewidth=2))
    axs[0].set_title('Selected correct pothole detection',fontsize=11); axs[0].axis('off')
    axs[1].imshow(im); hm=axs[1].imshow(heat,extent=[0,im.shape[1],im.shape[0],0],alpha=.6,cmap='RdYlBu_r',vmin=-max(abs(heat.min()),abs(heat.max())),vmax=max(abs(heat.min()),abs(heat.max())))
    axs[1].set_title('Confidence change when each cell is masked',fontsize=10); axs[1].axis('off')
    fig.colorbar(hm,ax=axs[1],fraction=.046,pad=.04,label='Baseline confidence minus masked confidence')
    figsave('occlusion.png')

pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold',r'C:\Windows\Fonts\arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyX',fontName='Arial',fontSize=9.5,leading=14,textColor=colors.HexColor(NAVY),spaceAfter=8))
styles.add(ParagraphStyle(name='SmallX',parent=styles['BodyX'],fontSize=8,leading=11,spaceAfter=5))
styles.add(ParagraphStyle(name='TitleX',fontName='Arial-Bold',fontSize=25,leading=30,textColor=colors.HexColor(NAVY),spaceAfter=15))
styles.add(ParagraphStyle(name='HeadX',fontName='Arial-Bold',fontSize=17,leading=21,textColor=colors.HexColor(NAVY),spaceAfter=12))
styles.add(ParagraphStyle(name='SubX',fontName='Arial-Bold',fontSize=11,leading=15,textColor=colors.HexColor(TEAL),spaceBefore=9,spaceAfter=6))
styles.add(ParagraphStyle(name='CellX',parent=styles['BodyX'],fontSize=8,leading=11,spaceAfter=0))
styles.add(ParagraphStyle(name='CellH',parent=styles['CellX'],textColor=colors.white,fontName='Arial-Bold'))
story=[]
def p(text,style='BodyX'): return Paragraph(text,styles[style])
def add(text,style='BodyX'): story.append(p(text,style))
def heading(k,title):
    if story: story.append(PageBreak())
    add(f'{k:02d}  /  {title}','HeadX')
def table(headers,rows,widths=None):
    data=[[p(str(c),'CellH') for c in headers]]+[[p(html.escape(str(c)),'CellX') for c in r] for r in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F5F7'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),1,colors.HexColor(TEAL))]))
    story.append(t); story.append(Spacer(1,10))
def picture(path,width=490):
    from PIL import Image as PILImage
    w,h=PILImage.open(path).size; story.append(Image(str(path),width=width,height=width*h/w)); story.append(Spacer(1,7))
def ci(k): return ' - '.join(pct(v) for v in D['bootstrap_95ci'][k])
W=A4[0]-96

# 1. Name and roll number are the first content on the first page.
add('Eswara Raj M','SubX'); add('Roll number: <b>CB.SC.U4CSE23022</b>')
story.append(Spacer(1,13)); add('Gopher-AI<br/>Model Test &amp; Evaluation','TitleX')
add('Framework-aligned evaluation report | 15 October 2026','SubX')
add('Evaluated checkpoint: <b>gopher-ai_v1.pt</b><br/>Task: detect five road-damage categories in dashcam images.')
add('This report contains fresh measurements from the supplied model and local RDD2022 data. It adapts the April 2024 <i>Test and Evaluation of Artificial Intelligence Models</i> framework to an object detector. The framework provides strategy-level guidance, not a fixed test recipe or universal pass score [1, printed pp. 3, 19, 41].')
table(['Primary result','Measured value'],[
    ['Test images / annotated objects',f"{num(B['images'])} / {num(A['splits']['test']['objects'])}"],
    ['mAP@0.50',pct(B['metrics']['metrics/mAP50(B)'])],
    ['mAP@0.50:0.95',pct(B['metrics']['metrics/mAP50-95(B)'])],
    ['Detection precision / recall at confidence 0.25',f"{pct(D['overall']['precision'])} / {pct(D['overall']['recall'])}"],
    ['Detection F1 at confidence 0.25',pct(D['overall']['f1'])],
    ['Image-level damage-presence accuracy',pct(D['image_level']['accuracy'])],
    ['Warm single-image latency, mean / p95',f"{X['latency']['mean_ms']:.2f} / {X['latency']['p95_ms']:.2f} ms"]],[W*.64,W*.36])
add('Evaluation outcome','SubX')
add(f"At the fixed operating threshold, the model correctly detected {num(D['overall']['tp'])} annotated objects and missed {num(D['overall']['fn'])}. These measurements support further model development and supervised review; they do not establish autonomous road-safety readiness. No stakeholder acceptance thresholds or live deployment measurements were supplied.")
add('All percentages are measured on the stated sample and configuration. Confidence intervals, subgroup differences, stress-test degradation, and untested areas are recorded in the following pages.','SmallX')

heading(2,'Test strategy and framework mapping')
add('The intended use is to assist a person reviewing road-surface images. Incorrect alarms create review work; missed damage can leave maintenance needs undetected. Model-level testing is performed here; a complete vehicle, application, camera, and operator workflow is outside this experiment [1, pp. 4, 56-57].')
table(['Framework area (printed page)','Implementation in this evaluation'],[
    ['Correctness (9, 45)','Full labeled test split; AP, fixed-threshold detection precision/recall/F1; secondary binary image accuracy and ROC-AUC.'],
    ['Bias / representativeness (11, 16)','Class counts, country-prefix composition, country-specific detection results, and train/test distribution differences.'],
    ['Robustness / test design (12, 20, 23)','Seeded 300-image sample; lighting, blur, noise, JPEG degradation; a complete 2 x 2 darkness-by-blur comparison.'],
    ['Metamorphic testing (22)','Horizontal flip with transformed ground-truth boxes and prediction consistency after reversing the flip.'],
    ['Uncertainty (14)','2,000 image-level bootstrap resamples for metric confidence intervals; confidence reliability and Brier score.'],
    ['Latency / resilience (17, 13)','100 warm batch-one timings, tail latency, memory; 20 clean-corrupt-clean recovery probes.'],
    ['Interpretability (10)','Local 6 x 6 occlusion sensitivity study of one correctly detected pothole.'],
    ['Data / documentation (31-37, 59-63)','Image/label audit; exact-duplicate split checks; checksums, manifests, model/data facts, scripts and machine-readable results.']],[W*.37,W*.63])
add('Controls and assumptions','SubX')
add('Weights were frozen throughout testing. No retraining or test-set threshold selection was performed. Confidence 0.25 and matching IoU 0.50 were fixed in advance for the operating-point analysis. AP uses scores down to 0.001 and IoUs 0.50 through 0.95. NMS IoU is 0.70 and maximum detections per image is 300.')
add('The primary split is the existing test folder. Filename and exact-content checks establish no observed exact overlap, but do not prove historical isolation: earlier evaluation outputs exist, and prior use for model selection cannot be ruled out. Near-duplicate scenes and route-level overlap were not assessed.','SmallX')

heading(3,'Data quality and representativeness')
table(['Split','Images','Objects*','Backgrounds','Background share'],[[s.title(),num(v['images']),num(v['objects']),num(v['backgrounds']),pct(v['backgrounds']/v['images'])] for s,v in A['splits'].items()],[70,80,85,90,W-325])
add('*Objects counted by this audit exclude a malformed zero-width training annotation. Duplicate annotation rows are counted and flagged; source labels were not edited.','SmallX')
table(['Class','Training objects','Validation objects','Test objects'],[[name,num(A['splits']['train']['classes'].get(str(i),0)),num(A['splits']['val']['classes'].get(str(i),0)),num(A['splits']['test']['classes'].get(str(i),0))] for i,name in enumerate(D['per_class'])],[W*.37,W*.21,W*.21,W*.21])
add('Audit findings','SubX')
add('All 5,076 test images decoded successfully and had corresponding label files; no invalid label rows, duplicate annotation rows, orphan labels, or exact duplicate test images were found. Pairwise split comparisons found zero shared filenames and zero shared SHA-256 image hashes.')
add('The training split has one zero-width annotation (Japan_001265, row 4), two images with duplicate annotation rows (Japan_006916 and Japan_011427), and one excess exact duplicate image. The validation split has one image with duplicate annotation rows (Japan_006536). These are data-quality findings, not automatic evidence of model failure.')
add('Representation limits','SubX')
add('The local test set contains country prefixes for Czech Republic, India, Japan, Norway, and United States. Country is inferred from filenames, not independently verified capture metadata. The repository documents filtering drone/motorbike images, retaining 1,000 training background images, and resizing training/validation images. This audit verifies the current files, not the historical execution of every preprocessing step.')
add(f"Training backgrounds comprise {pct(A['splits']['train']['backgrounds']/A['splits']['train']['images'])}, versus {pct(A['splits']['test']['backgrounds']/A['splits']['test']['images'])} in testing. This large sampling difference matters for false alarms. The data audit checks syntax and exact content; expert annotation correctness, privacy review, scene diversity, weather, night coverage, and deployment representativeness remain unverified.")

heading(4,'Correctness on the full test split')
add('Detection metrics are appropriate because the model must both localize an object and assign its damage category. For fixed-threshold results, predictions are processed from highest to lowest confidence and matched one-to-one to an unused same-class ground-truth box at IoU >= 0.50.')
table(['Class','GT','TP / FP / FN','Precision','Recall','F1'],[[n,num(D['per_class'][n]['tp']+D['per_class'][n]['fn']),f"{v['tp']} / {v['fp']} / {v['fn']}",pct(v['precision']),pct(v['recall']),pct(v['f1'])] for n,v in D['per_class'].items()]+[['All (micro)',num(A['splits']['test']['objects']),f"{D['overall']['tp']} / {D['overall']['fp']} / {D['overall']['fn']}",pct(D['overall']['precision']),pct(D['overall']['recall']),pct(D['overall']['f1'])]],[W*.25,W*.09,W*.24,W*.14,W*.14,W*.14])
add('Threshold-swept average precision','SubX')
aprows=[]
for r in B['per_class']:
    aprows.append([r.get('Class',r.get('class','')),pct(r.get('mAP50',0)),pct(r.get('mAP50-95',0))])
table(['Class','AP@0.50','AP@0.50:0.95'],aprows+[['Mean over five classes',pct(B['metrics']['metrics/mAP50(B)']),pct(B['metrics']['metrics/mAP50-95(B)'])]],[W*.50,W*.25,W*.25])
add('Precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = 2PR / (P + R). AP integrates the precision-recall curve; mAP averages class AP. AP@0.50:0.95 also averages ten localization thresholds. The Ultralytics AP calculation uses its built-in matching method and multi-label validation NMS; the fixed-threshold table uses the explicitly defined confidence-ordered matching rule.','SmallX')
add('Secondary image-level classification','SubX')
z=D['image_level']; add(f"Damage presence is positive if an image has any annotation; prediction is positive if any detection has confidence >= 0.25. TP={z['tp']:,}, FP={z['fp']:,}, TN={z['tn']:,}, FN={z['fn']:,}. Accuracy is <b>{pct(z['accuracy'])}</b>, ROC-AUC is <b>{z['roc_auc']:.4f}</b>, and specificity is {pct(z['specificity'])}. This task ignores location and damage type; it must not be presented as object-detection accuracy.")

heading(5,'Uncertainty and confidence reliability')
add('Confidence intervals describe sampling variability for the fixed model. Entire images, including their ground truths and detections, were resampled with replacement 2,000 times (seed 23022). The table reports the 2.5th and 97.5th percentiles. The model was not retrained during resampling.')
table(['Metric','Point estimate','95% image-bootstrap interval'],[
    ['Detection precision',pct(D['overall']['precision']),ci('precision')],
    ['Detection recall',pct(D['overall']['recall']),ci('recall')],
    ['Detection F1',pct(D['overall']['f1']),ci('f1')],
    ['Image-level accuracy',pct(z['accuracy']),ci('image_accuracy')]],[W*.36,W*.25,W*.39])
add('These intervals assume test images are exchangeable. Correlated frames or shared routes can make the intervals too narrow. They do not capture new-country uncertainty, labeling errors, model-training variability, or all aleatoric/epistemic uncertainty. This is evaluation bootstrapping, distinct from the training-resampling example in framework p. 23.','SmallX')
picture(FIG/'calibration.png',450)
c=D['detection_calibration']
add(f"For {num(c['predictions'])} retained detections, expected calibration error is <b>{c['ece_10_equal_width_bins']:.4f}</b> and detection Brier score is <b>{c['brier_score']:.4f}</b>. Ten equal-width bins span [0,1]; only occupied bins contribute. A detection is correct only if it matches a same-class object at IoU >= 0.50 under the stated matching rule.")
add('The 0.90-1.00 confidence bin contains only four detections, so its final downward point is highly uncertain. Bin sample counts are retained with the results; the chart should not be read as a smooth population calibration curve.','SmallX')
add(f"The binary image-level Brier score is {z['brier_score']:.4f}, using maximum detection confidence as a damage-presence score. That score is a proxy, not a separately trained image probability. No calibration was fitted on the test set. Low calibration error for retained detections does not account for missed objects or establish overall uncertainty quality.",'SmallX')

heading(6,'Subgroup performance and operating trade-offs')
add('Country-prefix groups provide a limited view of representation bias. The table uses the same confidence and matching thresholds as the primary fixed-threshold analysis. Differences may reflect class mix, image conditions, annotation practices, or geographic domain effects; they do not establish a causal explanation or demographic fairness.')
table(['Country prefix','Images','Objects','Precision','Recall','F1'],[[{'Czech':'Czech Republic','United':'United States'}.get(n,n),num(v['images']),num(v['objects']),pct(v['precision']),pct(v['recall']),pct(v['f1'])] for n,v in D['subgroups'].items()],[W*.25,W*.12,W*.15,W*.16,W*.16,W*.16])
picture(FIG/'subgroups.png',460)
gap=max(v['recall'] for v in D['subgroups'].values())-min(v['recall'] for v in D['subgroups'].values())
add(f'The highest-to-lowest country recall gap is <b>{gap*100:.2f} percentage points</b>. All groups contain multiple images, but coverage of each class within each environment is not guaranteed. No protected-person attributes are available.')
add('Confidence-threshold sensitivity','SubX')
table(['Confidence','Precision','Recall','F1','FP / FN'],[[f"{v['confidence']:.2f}",pct(v['precision']),pct(v['recall']),pct(v['f1']),f"{v['fp']:,} / {v['fn']:,}"] for v in D['threshold_sweep']],[75,90,90,90,W-345])
add('This sweep illustrates trade-offs; it was not used to select a best test threshold. Any deployment threshold should be chosen on separate validation data using agreed costs of misses and false alarms.','SmallX')

heading(7,'Robustness and metamorphic testing')
add('A uniform random sample of 300 test images (seed 23022, without replacement) was reused for every condition. Images were resized to a 512-pixel longest side before corruption, then square-letterboxed for inference. Each condition uses the same ground truth, except horizontal-flip boxes are mirrored. Results below are micro detection metrics at confidence 0.25 / IoU 0.50.')
labels={'clean':'Clean reference','dark_0.75':'Brightness x0.75','dark_0.50':'Brightness x0.50','bright_1.50':'Brightness x1.50 (clipped)','gaussian_blur_5':'Gaussian blur: 5x5, sigma 1.5','gaussian_noise_15':'Gaussian noise: sigma 15/255','jpeg_quality_30':'JPEG quality 30','dark_0.50_and_blur_5':'Brightness x0.50 + blur','horizontal_flip':'Horizontal flip'}
table(['Condition','Precision','Recall','F1','F1 change (pp)'],[[labels[n],pct(v['precision']),pct(v['recall']),pct(v['f1']),f"{v.get('f1_delta_percentage_points',0):+.2f}"] for n,v in X['corruptions'].items()],[W*.40,W*.15,W*.15,W*.15,W*.15])
add(f"The sample contains {X['corruptions']['clean']['objects']:,} objects. Realized country counts: "+', '.join(f'{k} {v}' for k,v in X['sample_countries'].items())+'. Its clean result is a separate sample/configuration baseline, not directly interchangeable with the full-set AP run. Predict-mode uses single-label NMS.','SmallX')
add('Pairwise interaction test','SubX')
add('Clean, darkness-only, blur-only, and darkness-plus-blur cover all four combinations of two binary factors. This is a complete 2 x 2 factorial test for these factors; it does not cover every pair among all possible environmental variables. The simulated corruptions probe input sensitivity and are not measurements of real night, rain, or fog performance.')
add('Metamorphic relation: horizontal reflection','SubX')
mt=X['metamorphic']; add(f"After flipping predicted boxes back, {mt['matched_boxes']:,} same-class boxes match the clean predictions at IoU >= 0.50. Clean-detection retention is <b>{pct(mt['clean_detection_retention'])}</b>, symmetric box-consistency F1 is <b>{pct(mt['symmetric_consistency_f1'])}</b>, and {mt['images_with_complete_consistency']}/{mt['images']} images have complete box consistency (including images empty in both runs).")
add('The proposed relation is approximate invariance to reflection. It is a diagnostic assumption, not a guarantee that every road scene is semantically identical when mirrored. No adaptive adversarial attack, poisoning experiment, or red-team exercise was performed.','SmallX')

heading(8,'Latency and recovery from perturbation')
l=X['latency']; r=X['resilience']
add(f"Measurements used {M['device']} ({M['gpu_memory_bytes']/1024**3:.0f} GiB), FP32, 512 x 512 input, and four PyTorch CPU threads. Batch-one timings used 10 warm-up images followed by 100 images with CUDA synchronization around each complete prediction call. Confidence was 0.25. The timer includes API overhead, preprocessing, inference, and postprocessing; file decode, initial resize, camera capture, network transport, and UI rendering are excluded.")
table(['Latency / resource measure','Measured value'],[
    ['Mean / median prediction-call latency',f"{l['mean_ms']:.2f} / {l['median_ms']:.2f} ms"],
    ['95th / 99th percentile latency',f"{l['p95_ms']:.2f} / {l['p99_ms']:.2f} ms"],
    ['Minimum / maximum latency',f"{l['min_ms']:.2f} / {l['max_ms']:.2f} ms"],
    ['Serial throughput (1000 / mean milliseconds)',f"{l['serial_images_per_second']:.2f} images/s"],
    ['Mean internal preprocess time',f"{l['mean_stages_ms']['preprocess']:.2f} ms"],
    ['Mean internal inference time',f"{l['mean_stages_ms']['inference']:.2f} ms"],
    ['Mean internal postprocess time',f"{l['mean_stages_ms']['postprocess']:.2f} ms"],
    ['Peak CUDA tensor allocation during timed sample',f"{l['peak_cuda_allocated_bytes']/1024**2:.2f} MiB"]],[W*.70,W*.30])
add(f"The full test validator separately reported {B['speed_ms']['inference']:.2f} ms/image of inference with batch size 8. This is an amortized component time, not single-image end-to-end latency. GPU tensor allocation excludes reserved allocator memory and other processes; it is not total device memory use.",'SmallX')
add('Clean-corrupt-clean resilience probe','SubX')
add(f"Twenty input sequences applied the original image, a darkened version (brightness x0.50), then the original image again. <b>{r['restored_output_trials']}/{r['trials']}</b> restored predictions matched their initial outputs (class list exact, confidence absolute tolerance 1e-6, box tolerance 1e-4; NumPy default relative tolerance 1e-5). The mean restoration-check time, including the prediction call, output conversion and comparison, was <b>{r['mean_recovery_call_ms']:.2f} ms</b>.")
add('This is a stateless feed-forward detector, so the probe tests lingering input effects in the inference process. It does not test crash recovery, sensor reconnection, tracking state, service failover, power interruption, or recovery of a complete deployed system. No operational latency requirement was supplied, so no real-time pass/fail claim is made.')

heading(9,'Local interpretability probe')
if 'interpretability' in X:
    it=X['interpretability']; add('A post-hoc, model-agnostic occlusion study was applied to one selected correct pothole detection. The image was partitioned into a 6 x 6 grid. One cell at a time was replaced with gray value 114, and the model was rerun. The tracked score is the highest pothole confidence at IoU >= 0.50 with the original predicted box; it becomes zero when no such detection remains.')
    picture(FIG/'occlusion.png',W)
    table(['Probe fact','Measured result'],[
        ['Image',it['image']],['Selection rule',it['selection']],['Baseline confidence',f"{it['baseline_confidence']:.4f}"],
        ['Number of occlusion runs','36'],['Maximum confidence decrease',f"{it['max_confidence_drop']:.4f}"],
        ['Mean signed confidence decrease',f"{it['mean_confidence_drop']:.4f}"],
        ['Minimum signed confidence decrease',f"{it['min_confidence_drop']:.4f}"]],[W*.44,W*.56])
    add('Positive values identify cells whose removal reduced the tracked confidence; negative values identify cells whose removal increased it. A large drop can reflect changed localization as well as changed class confidence. Gray occlusion itself can create an out-of-distribution artifact.')
    add('This probe shows local sensitivity, not a causal explanation of learned reasoning. It is neither a global explanation nor proof that the model exclusively uses road-damage features. The selection rule favors a strong correct detection and is not a random estimate of explanation quality. No human explanation-usability study was performed.')
else: add('No eligible correct pothole detection was found for the defined local probe; interpretability was not measured.')

heading(10,'Interpretation, limitations and next actions')
worst=min(X['corruptions'].items(),key=lambda x:x[1]['f1']); low=min(D['subgroups'].items(),key=lambda x:x[1]['recall'])
add(f"The model's full-test mAP@0.50 is {pct(B['metrics']['metrics/mAP50(B)'])}, while mAP@0.50:0.95 is {pct(B['metrics']['metrics/mAP50-95(B)'])}. The stricter localization metric is lower, so detecting an approximate region does not imply precise bounding boxes. At confidence 0.25, {pct(1-D['overall']['recall'])} of annotated objects are missed and {pct(z['background_false_alarm_rate'])} of background-only images receive at least one alarm.")
add(f"The lowest measured subgroup recall is {pct(low[1]['recall'])} for the {low[0]} prefix. The lowest stress-sample F1 is {pct(worst[1]['f1'])} under {labels[worst[0]]}. These are priorities for investigation, not proof that one geographic or environmental factor causes the errors.")
table(['Framework method / concern','Status and reason'],[
    ['Holdout validation','Executed on the existing test split; historical test isolation remains unproven.'],
    ['Cross-validation / training bootstrap','Not executed. Valid evaluation across training folds requires retraining independent models; subdividing this model\'s test set would not provide that evidence.'],
    ['A/B or back-to-back model comparison','Not executed. No agreed task-compatible comparator was selected; the COCO-pretrained model has different labels.'],
    ['Adversarial testing / red teaming','Not executed. Natural corruption probes do not establish resistance to optimized attacks or data poisoning.'],
    ['Temporal drift','Not measured. No timestamped production series or newer labeled deployment data was supplied. Train/test composition differences are not temporal drift.'],
    ['Fairness, field testing and compliance','Not established. Requires relevant stakeholder definitions, metadata, deployment evidence, and review.'],
    ['Continuous monitoring / human evaluation','Not established. This is a discrete automated model-level test event, with a local sensitivity visualization.']],[W*.36,W*.64])
add('Recommended follow-up','SubX')
add('1. Review missed potholes and weak country/class combinations; repair verified annotation defects in versioned data.<br/>2. Choose operating thresholds on separate validation data using agreed miss/alarm costs; preserve an untouched future test set.<br/>3. Collect route-disjoint, labeled deployment images covering intended cameras, roads, weather and light; include verified negative scenes.<br/>4. Re-evaluate new versions, add adversarial and system recovery tests, and monitor field latency, class mix and labeled performance over time.')

heading(11,'Model card, reproducibility and references')
table(['Model / data fact','Recorded value'],[
    ['Owner / report identity','Eswara Raj M | CB.SC.U4CSE23022'],
    ['Architecture / learning paradigm','YOLO11n; supervised convolutional object detection; five classes'],
    ['Input / output','Color images; bounding boxes, damage classes and confidence scores'],
    ['Unfused parameters / checkpoint size',f"{M['parameter_count']:,} / {M['model_file_bytes']:,} bytes"],
    ['Checkpoint metadata',f"Epoch {M['checkpoint_epoch_zero_based']} (zero-based; epoch {M['checkpoint_epoch_zero_based']+1}); date {M['checkpoint_date']}"],
    ['Recorded training configuration','AdamW; initial LR 0.0015; image size 512; batch 16; patience 10; maximum 100 epochs; seed 0; AMP disabled'],
    ['Evaluation software',f"Python {M['python'].split()[0]}; PyTorch {M['torch']}; Ultralytics {M['ultralytics']}"],
    ['Source and intended use','Local RDD2022 derivative; academic road-damage analysis with human review'],
    ['Rights and provenance limit','Repository describes academic/research use; dataset license, consent and full chain of custody were not independently verified.']],[W*.36,W*.64])
add('Version identifiers','SubX')
add('Checkpoint SHA-256:<br/>'+A['model_sha256'],'SmallX')
add('Ordered image-and-label manifest SHA-256:<br/>'+A['manifest_sha256'],'SmallX')
add('Reproduce the measurements','SubX')
add('Run the included scripts from the repository: audit_data.py, run_baseline.py, analyze_results.py, then run_extended.py. They use the existing weights and local dataset. The evidence bundle includes the scripts, settings, scalar metrics, per-image results, audit findings and hash manifest. Raw predictions and complete dataset records are retained in output/evaluation in the workspace. Report generation uses the bundled PDF environment.','SmallX')
add('References and page convention','SubX')
add('[1] CDAO Assessment and Assurance. <i>Test and Evaluation of Artificial Intelligence Models: What to Consider in a Test &amp; Evaluation Strategy.</i> April 2024. User-supplied PDF, 66 PDF pages. All cited page numbers in this report refer to printed slide numbers; PDF page = printed page + 1. This report paraphrases and adapts the framework.','SmallX')
add('[2] Local project evidence: README.md, new_train.py, runs/detect/train-10/args.yaml, gopher-ai_v1.pt checkpoint metadata, and current RDD2022 files. The supplied checkpoint matches the local best.pt SHA-256. Training configuration is recorded metadata, not independently reproduced training.','SmallX')
add('[3] Installed Ultralytics 8.4.90 DetectionValidator and detection metrics implementation, inspected locally; fresh execution results and evaluation scripts accompanying this report.','SmallX')

def decorate(c,doc):
    c.saveState(); width,height=A4
    c.setStrokeColor(colors.HexColor(TEAL)); c.setLineWidth(2); c.line(48,height-35,width-48,height-35)
    c.setFont('Arial',8); c.setFillColor(colors.HexColor(MUTED))
    c.drawString(48,29,'GOPHER-AI  |  Eswara Raj M  |  CB.SC.U4CSE23022')
    c.drawRightString(width-48,29,f'{doc.page}'); c.restoreState()

pdf=O/'Gopher_AI_Test_and_Evaluation_Eswara_Raj_M.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=53,bottomMargin=48,
    title='Gopher-AI Model Test and Evaluation - Eswara Raj M',author='Eswara Raj M',subject='Measured framework-aligned evaluation of gopher-ai_v1.pt')
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
readme='''Gopher-AI evaluation evidence\nIdentity: Eswara Raj M, CB.SC.U4CSE23022\nEvaluation date: 15 October 2026\n\nRun from the Gopher-AI repository with the existing dataset and checkpoint:\n  python evaluation/audit_data.py\n  python evaluation/run_baseline.py\n  python evaluation/analyze_results.py\n  python evaluation/run_extended.py\n\nEnvironment: Python 3.12.10, torch 2.11.0+cu128, ultralytics 8.4.90, NVIDIA GTX 1650.\nReport generation additionally requires reportlab, matplotlib, numpy and Pillow, and Arial fonts.\nRaw baseline/stress predictions and detailed source records remain in output/evaluation.\nSource data and weights are excluded from this bundle.\nThe existing test set is not certified historically untouched.\nNo deployment acceptance thresholds were supplied.\n'''
(E/'README.txt').write_text(readme)
bundle=ROOT/'output/Gopher_AI_Evaluation_Evidence.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as zf:
    for f in (ROOT/'evaluation').glob('*.py'): zf.write(f,f.relative_to(ROOT))
    for name in ['audit.json','model_metadata.json','baseline_metrics.json','analysis.json','extended_results.json','per_image_results.csv','data_manifest.txt','data.yaml','README.txt']:
        zf.write(E/name,'results/'+name)
    for f in FIG.glob('*.png'): zf.write(f,'figures/'+f.name)
print(pdf); print(bundle)
