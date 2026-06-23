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
