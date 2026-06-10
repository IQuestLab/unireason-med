# Data

The UniReason-Med training data is hosted on Hugging Face:

```text
https://huggingface.co/datasets/IQuestLab/UniReason-Med-Data
```

## Overview

The paper introduces UniMed-CoT, a medical multimodal instruction-tuning dataset for UniReason-Med. UniReason-Med studies a shared grounded reasoning interface for medical VQA across 2D images and 3D volumes. The model is trained to interleave textual reasoning with localized visual evidence using shared bounding-box syntax and region-token injection, so the same policy can operate on either a 2D image or a slice-serialized 3D volume.

UniMed-CoT contains 220K grounded chain-of-thought samples: 170K 2D samples and 50K 3D samples. The paper builds the data from SAMed2D-v1 for 2D medical images and M3D training cases for 3D volumetric CTs. Grounding coordinates are extracted from segmentation masks, QA pairs are generated, and GPT-4o is used to produce interleaved grounded CoT annotations.

The released dataset provides the data used by the SFT and GRPO/RL training pipelines in this repository. It is packaged in Parquet format with separate splits for SFT records, RL records, and released 2D image bytes.

## 3D Data Notice

The public Hugging Face release keeps 3D examples text-only and does not redistribute 3D image data. These samples are derived from M3D, whose underlying image sources include Radiopaedia and may require separate authorization. Users who need the original 3D images should obtain the necessary permission from the original data providers, including M3D/Radiopaedia where applicable.

## Images and Paths

Released 2D images are stored in the dataset image split and referenced from SFT/RL records by neutral image identifiers. Local materialization scripts write image files and keep image references as relative paths, avoiding absolute user or machine paths in generated training files.
