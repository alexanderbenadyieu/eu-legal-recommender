# Multi-Tier Summarization Strategy for Legal Documents

This document outlines a multi-tier summarization approach designed to handle legal texts of varying lengths. Each tier employs a different strategy based on the document length to ensure that key legal information is preserved while producing a concise summary. The tiers are defined as follows:


- **Tier 1 (0–600 Words):**  
  **Single-Step Abstractive Summarization.**  
  The full text is fed directly into an abstractive model (e.g., BART) using adaptive compression ratios.

- **Tier 2 (600–2,500 Words):**  
  **Two-Step Summarization.**  
  First, an extractive step selects salient content (using a rule like K = max(300, min(0.3×D, 600))). Then, an abstractive model generates the final summary.

- **Tier 3 (2,500–20,000 Words):**  
  **Hierarchical Summarization (Up to 50th Percentile).**  
  The document is split into 200–300 word chunks with fixed extraction percentages (higher for earlier chunks), aggregated to form Lₑ, and then refined through one or more dependent extraction passes until reduced to about 1,500 words. Finally, chained abstractive summarization (using BART) produces a summary of roughly 480–600 words.

- **Tier 4 (20,000–68,000 Words):**  
  **Hierarchical Summarization for Mid-Ultra-Long Documents.**  
  A fixed-ratio extraction (with slightly lower percentages) is applied to 200–300 word chunks to form Lₑ, which is then iteratively refined via dependent extraction to about 2,500 words. LongformerBART is then used to generate a final summary of approximately 600–800 words.

- **Tier 5 (68,000–150,000 Words; 90th–95th Percentile):**  
  **Ultra-Long Document Summarization.**  
  The document is segmented using its inherent structure (e.g., chapters/sections), with fixed extraction applied per section (using lower rates for later sections). Multiple dependent extraction passes reduce the global Lₑ to about 3,000–4,000 words, which is then summarized with a chained abstractive process (using LongformerBART/LED) into a final summary of 600–800 words.

- **Tier 6 (150,000–500,000 Words; Top 5%):**  
  **Ultra-Ultra-Long Document Summarization.**  
  The document is hierarchically segmented into major sections and sub-sections. Extraction is applied at the sub-section level (favoring early content) and aggregated into chapter summaries. Several rounds of dependent extraction reduce the overall text to about 4,000–5,000 words, and an extended-context abstractive summarization (using LongformerBART/LED with chained passes) produces a final summary of 600–800 words.
---

## Tier 1: Single-Step Abstractive Summarization (0–600 Words)

### Process Description

For documents up to 600 words, the entire text can be fed directly into an abstractive summarization model (e.g., BART). An adaptive ratio formula is applied to determine the target summary length, ensuring a range (minimum and maximum word count) rather than a fixed output.

### Adaptive Ratio Specifications

| **Document Length** | **Target Summary Range** | **Approximate Compression Ratio**       |
|---------------------|--------------------------|-----------------------------------------|
| 0–150 words         | 15–50 words              | 33%–100% (i.e., summary is 33–100% of the original) |
| 151–300 words       | 50–100 words             | 17%–33%                                |
| 301–600 words       | 100–200 words            | 17%–33%                                |

**Notes:**
- The ratio ensures that very short texts receive a sufficiently detailed summary.
- Longer texts in this range are compressed proportionally while still preserving essential information.

---

## Tier 2: Two-Step Summarization (600–2,500 Words)

### Overview

Documents in the 600–2,500 word range require an initial extractive step to reduce the text size, followed by an abstractive summarization of the extracted content. This two-step process ensures that the summarization model receives an input that is both focused and within manageable length limits.

### Step 1: Extractive Summarization

**Objective:**  
Select the most salient content from the full document.

**Process:**
- **Input:** Full document (D words).
- **Target Extraction (K):**  
  Use the rule: K = max(300, min(0.3 × D, 600))


This ensures:
- A minimum of 300 words is always extracted.
- For longer texts, extraction is capped at 600 words.
- Approximately 30% of the document is extracted if that value lies within 300–600 words.

### Step 2: Abstractive Summarization

**Objective:**  
Generate a final summary from the extracted content.

**Process:**
- **Input:** The extracted text (K words) from Step 1.
- **Target Summary (S):**  
Apply an additional compression: S = between 0.6 × K and 0.8 × K words


For example:
- A 300-word extraction yields a final summary of approximately 180–240 words.
- A 600-word extraction yields a final summary of approximately 360–480 words.

### Adaptive Ratios Table (Examples)

| Document Length (D) | Extracted Content (K) | Final Summary (S)  | Approximate Final Ratio (S/D) |
|---------------------|-----------------------|--------------------|-------------------------------|
| 600 words           | 300 words             | 180–240 words      | 30%–40%                       |
| 1,000 words         | 300 words             | 180–240 words      | 18%–24%                       |
| 1,500 words         | 450 words             | 270–360 words      | 18%–24%                       |
| 2,000 words         | 600 words             | 360–480 words      | 18%–24%                       |
| 2,500 words         | 600 words             | 360–480 words      | 14%–19%                       |

**Notes:**
- For shorter documents, a higher proportion of content is preserved.
- For longer documents in this tier, the final summary represents a smaller percentage of the original text, ensuring conciseness.

---


# Tier 3: Hierarchical Summarization for Documents up to 50th percentile in length(2,500–20,000 Words)

For documents in the Tier 3 range, our goal is to ultimately feed a text of around 1,500 words into the final abstractive summarizer (BART) so that we can generate a final summary of 480–600 words. To achieve this, we employ a multi-step process that involves:

1. An initial fixed-ratio extractive stage (Step 1)
2. One or more additional dependent-ratio extractive steps (Steps 2 and possibly 3) to further reduce the aggregated extraction to about 1,500 words
3. A final abstractive summarization stage using chained summarization to reach the target output length

Below is a detailed description of Steps 2 and 3 and the final abstractive summarization.

---

## Step 1: Fixed-Ratio Extractive Stage (Pre-defined)

- **Segmentation:**  
  Split the document into fixed-size chunks (approximately 200–300 words each).

- **Per-Chunk Extraction:**  
  Extract a fixed percentage from each chunk, with higher percentages from earlier chunks:
  - 1st chunk: ~34%
  - 2nd chunk: ~30%
  - 3rd chunk: ~24–25%
  - 4th chunk: ~20%
  - 5th chunk: ~16–17%
  - Any chunk beyond the fifth: ~12.5%

- **Aggregation:**  
  Concatenate the extracted portions from all chunks to form the aggregated extraction, denoted as Lₑ.

---

## Step 2: Dependent-Ratio Extractive Refinement

Since Lₑ from Step 1 will likely be too long to directly feed into BART (given its context limitations), we need to further reduce Lₑ to approximately 1,500 words. We do this with an additional extraction pass that applies a dependent ratio across the aggregated text.

### Mathematical Logic for Step 2

1. **Define the Target Length (T):**  
   Set T = 1,500 words (the desired length for the final extractive output).

2. **Compute the Compression Factor (f):**  
   Calculate the factor as: f = T / Lₑ


This factor tells you what fraction of Lₑ you want to retain.

3. **Enforce a Minimum Extraction Rate:**  
Since we want to ensure that we do not extract too little from any segment, we clamp f to a minimum value. For example, if f falls below 15% (i.e. 0.15), then set f = 0.15. This prevents overly aggressive compression in a single step.

4. **Apply the Extraction:**  
The output of Step 2, denoted as L₂, is given by: L₂ = f × Lₑ


5. **Decision for Additional Extraction (Step 3):**  
- If L₂ is still significantly above 1,500 words, then a second dependent extraction pass (Step 3) will be applied.
- Otherwise, proceed to the final abstractive summarization stage.

---

## Step 3: (Optional) Additional Dependent Extraction

If after Step 2 the extracted text L₂ remains too long for comfortable processing by BART (i.e., it is still much longer than 1,500 words), we perform a similar dependent extraction again:

1. **Recompute the Compression Factor:**  
Let T still equal 1,500 words, and now compute: f' = T / L₂


2. **Clamp f' to a Minimum:**  
Again, ensure that f' is not less than 0.15 (or another minimum rate based on empirical testing).

3. **Apply the Extraction:**  
The output L₃ is then: L₃ = f' × L₂


4. **Iteration:**  
Repeat this step until the output text is in the vicinity of 1,500 words. Typically, this process will require 1–3 dependent extraction passes, depending on the original document length.

---

## Final Abstractive Summarization: Chained Summarization

Once the extractive refinement yields a text (denoted L_final) of around 1,500 words, we proceed to the final abstractive summarization using BART. Because BART’s effective context window is around 600–800 words, we may need to use chained summarization:

1. **First Abstractive Pass:**  
- Input: L_final (≈1,500 words) 
- Objective: Summarize L_final into an intermediate summary of approximately 600–800 words using Longformer BART
- This step ensures that the input is further reduced into a form that can be processed in one pass by BART.

2. **Second Abstractive Pass:**  
- Input: The intermediate summary (≈600–800 words)
- Objective: Generate the final summary with the desired length of 480–600 words.
- Set BART decoding parameters (min_length and max_length) to enforce the target summary length.

---


# Tier 4: Hierarchical Summarization for Documents in the 50th to 75th percentiles(20,000–68,000 Words)

For documents in the 20,000–68,000 word range, a more aggressive and multi-step hierarchical approach is required to reduce the content while preserving key legal information. In Tier 4, our aim is to reduce the document via multiple extractive passes until we obtain a refined text of about 2,500 words, which can then be fed into a final abstractive summarization stage using LongformerBART to generate a summary of 600–800 words.

---

## Step 1: Fixed-Ratio Extractive Stage

- **Segmentation:**  
  Divide the document into fixed-size chunks (approximately 200–300 words each).

- **Per-Chunk Extraction:**  
  Apply fixed extraction percentages to each chunk, with higher percentages from the earlier chunks:
  - 1st chunk: 27–33% 
  - 2nd chunk: 22–27%
  - 3rd chunk: 18–22% 
  - 4th chunk: 15–18% 
  - 5th chunk: 15–18% 
  - 6th chunk: 12.5–15% 
  - Any chunks beyond the sixth: 8–12.5% (e.g., 12.5%)

- **Aggregation:**  
  Concatenate the extracted portions from all chunks to form an aggregated extraction, denoted as Lₑ.

---

## Step 2: Dependent-Ratio Extractive Refinement

Due to the much longer input, Lₑ will be considerably longer than our target length for the final extraction. We now apply additional extraction passes to reduce Lₑ to approximately 2,500 words.

- **Define Target Length:**  
  Set T = 2,500 words.

- **Compute Compression Factor:**  
  Calculate the factor f = T / Lₑ.  
  If f is less than 0.15 (15%), clamp it to 0.15 to avoid overly aggressive compression.

- **Apply Dependent Extraction:**  
  Extract f × Lₑ words from the aggregated text. This yields a refined extraction, L₂.

- **Iterate if Needed:**  
  If L₂ is still significantly longer than 2,500 words, repeat the dependent extraction:
  - Recompute f' = T / L₂ (again clamping to a minimum of 15%).
  - Extract L₃ = f' × L₂.
  - Continue this process until the final extractive output (L_final) is approximately 2,500 words.

---

## Step 3: Final Abstractive Summarization: LongformerBART

Once the dependent extraction yields a text L_final of around 2,500 words, we perform the final summarization using LongformerBART.
   - **Input:** L_final (≈2,500 words)  
   - **Objective:** Use LongformerBART to condense L_final into a summary of approximately 600–800 words.  


STILL NOT DEFINED:
## Tier 5: Ultra-Long Documents (68,000–150,000 Words; 90th to 95th Percentile)
## Tier 6: Ultra-Ultra-Long Documents (150,000–500,000 Words; 95th to 100th Longest)
