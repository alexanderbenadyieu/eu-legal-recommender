# Multi-Tier Summarization Strategy for Legal Documents

This document outlines a multi-tier summarization approach designed to handle legal texts of varying lengths. Each tier employs a different strategy based on the document length to ensure that key legal information is preserved while producing a concise summary. The tiers are defined as follows:

- **Tier 1 (0–600 Words):**  
  **Single-Step Abstractive Summarization.**  
  The full text is fed directly into an abstractive model (e.g., BART) using adaptive compression ratios.

- **Tier 2 (600–2,500 Words):**  
  **Two-Step Summarization.**  
  First, an extractive step selects salient content (using a rule like K = max(300, min(0.3×D, 600))). Then, an abstractive model generates the final summary.

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