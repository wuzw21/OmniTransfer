# Related Work

## Widget Mapping And Test Migration

TEMdroid and SemFinder frame UI transfer as widget matching across app versions
or platforms. They are close to the target problem, but their stronger variants
often rely on heavier semantic encoders or metadata-only assumptions.

## Feature Matching

SuperGlue and LightGlue provide the architectural analogy: local descriptors
are contextually matched through a learned assignment mechanism. OmniTransfer
borrows the matching framing but uses structured UI graph features instead of
image keypoints as the primary hot-path representation.

LoFTR and MatchFormer further support conditioning descriptors on both inputs
through interleaved self- and cross-attention. GlueStick shows how explicit
connectivity can be injected into learned matching. These are architecture
references, not same-task GUI baselines.

## UI Representation Pretraining

UIBert, ActionBERT, and Screen2Vec show that UI components benefit from learned
context, position, and interaction signals. MobileViews and RICO provide the
unlabeled screenshot/view-hierarchy scale needed for self-supervised graph
pretraining. OmniTransfer uses these sources to learn a small matcher rather
than deploying their heavier encoders in the replay hot path.

AMEX adds modern high-resolution Android screenshots, XML, element functions,
and long action chains. AITW provides much larger screenshot/action coverage but
lacks a view hierarchy, so it is useful for visual and verified-outcome signals
rather than direct widget-pair gold.

Uni-GUI-OpenMobile adds a newer AndroidWorld-derived source with about 25.9K
screens from 2,640 trajectories over 19 open-source apps. Unlike coordinate-only
mobile trajectories, its per-step metadata includes bounded UI elements and
attributes, so it can enter the same self-supervised UIGraph pretraining path.
Action boxes remain auxiliary labels and do not define an inference shortcut.

OS-Atlas contributes a second AndroidWorld-derived grounding source with 15,905
screens and 89,860 bounded elements. Its public mobile annotations group all
elements by screenshot, which makes independent same-screen views and learned
NULL targets available without inventing cross-screen correspondences. The
larger AMEX portion is storage-heavy; OmniTransfer records it as a future source
rather than silently dropping its visual modality.

AndroidControl provides 15,283 human demonstrations over 833 apps, with both
screenshots and accessibility trees. OmniTransfer streams each official GZIP
TFRecord shard directly into resized UI graphs, so the 49.93 GB raw snapshot
does not need to persist on 9207. Official train, validation, test, IDD,
task-unseen, app-unseen, and category-unseen episode partitions are preserved.
AITW and GUI-Odyssey are still treated as trajectory or outcome sources because
click coordinates alone are not verified node-pair labels. For GUI-Odyssey,
OmniTransfer can distill the released SAM2 action boxes into weak pseudo UI
graphs for representation pretraining, while preserving the same learned
matcher path and keeping the boxes out of runtime matching.

WebUI contributes substantially larger cross-viewport web coverage. Its
Web-70k element release contains 173,546 screenshot rows with semantic element
boxes, while the official collection groups pages by domain before creating
train, validation, and test partitions. OmniTransfer uses the boxes only to
form visual graph nodes; correspondence and NULL decisions remain learned by
the same cross-attention matcher. Official validation and test partitions stay
frozen and never enter pretraining.

## Web And WebView

Mind2Web supplies DOM candidates, including positive and negative action
candidates, across sites and domains. Multimodal-Mind2Web adds screenshots and
raw traces. WebLINX provides HTML, screenshots, and multi-turn demonstrations;
BrowserGym offers a common observation interface. WebSight provides synthetic
HTML-screenshot scale for viewport, theme, locale, and CSS-reflow augmentation.
ReproBreak contributes reproduced locator changes across repository versions.

These sources are normalized into the same UI graph. For WebView, DOM or AX
nodes are attached below a bounded native WebView host and are matched by the
same cross-attention core. VisualWebArena and ScreenSpot-Pro remain frozen
evaluation sources rather than training corpora.

Detailed source verification, architecture mapping, and experiment protocol are
recorded in `docs/relation_aware_cross_attention_matcher.md`.

## GUI Grounding

Recent GUI-agent work such as ShowUI, UI-TARS, GUI-R1, UI-R1, ScreenSpot-Pro,
and coordinate-free grounding systems studies full visual grounding or action
policies. OmniTransfer is narrower: it is a relocation component inside a cached
GUI-agent replay system.

## UI Parsing

OmniParser-style screen parsing and Screen2AX-style accessibility hierarchy
reconstruction support the idea that UI should be represented as structured
tokens and relations. OmniTransfer can consume such tokens, but does not require
a heavy parser in the on-device hot path.
