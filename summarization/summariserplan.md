# Multi-Tier Summarization Strategy for Legal Documents

This document outlines a multi-tier summarization approach designed to handle legal texts of varying lengths. Each tier employs a different strategy based on the document length to ensure that key legal information is preserved while producing a concise summary. The tiers are defined as follows:

- **Tier 1 (0–600 Words):**  
  **Single-Step Abstractive Summarization.**  
  The full text is fed directly into an abstractive model (BART) using adaptive compression ratios.

- **Tier 2 (600–2,500 Words):**  
  **Two-Step Summarization.**  
  First, an extractive step selects salient content (using a rule like K = max(300, min(0.3×D, 600))). Then, an abstractive model generates the final summary.

- **Tier 3 (2,500–20,000 Words):**  
  **Hierarchical Summarization with Section-Based Extraction.**  
  Each section is processed as follows: sections under 350 words are used as is; sections from 350–750 words are divided into ≤350‑word chunks and reduced to about 350 words; sections between 750–1500 words are processed as a whole to ~350 words; and sections over 1500 words are subdivided into multiple chunks (each ultimately reduced to ≤350 words). These section summaries are aggregated into a global text (Lₑ), which is then refined via a weighted, dependent extraction (using a computed compression factor) to around 600–750 words. Finally, baseline BART is used to generate a final summary of 480–600 words.

- **Tier 4 (20,000–117,000 Words):**  
  **Section-Level Pre-Summarization with Global Dependent Extraction.**  
  Step 1 - Section-Based Pre-Summarization: For each section, different approaches are used based on length: (a) sections under 750 words get direct BART summarization to ≤350 words; (b) sections between 750–1500 words are first reduced extractively to ~600 words then summarized by BART to ≤350 words; (c) sections over 1500 words are split into subsections (max 1500 words), each subsection is extracted to ~600 words and then summarized by BART to ≤350 words. Step 2 - Global Dependent Extraction: All section summaries are combined and compressed using weighted extraction (weights decrease from 1.2 for first chunk to 0.5 for later chunks) with a compression factor f = target(750)/total_words (minimum 0.15), clamped to 15-35% extraction per chunk. Step 3 - Final Abstractive Summarization: The ~750-word extraction is summarized by BART to produce the final summary of 480–600 words.

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

# Tier 3: Hierarchical Summarization for Documents (2,500–20,000 Words)

In Tier 3, our objective is to reduce a long legal document to an intermediate extractive output of roughly 600–750 words. This intermediate text is then fed into an abstractive summarizer to produce a final summary of 480–600 words. The process is section-aware and uses both fixed and dependent extraction steps with mathematically defined compression factors.

---

## Step 1: Section-Based Fixed-Ratio Extraction

We first analyze the document by its sections and process each section as follows:

### A. Processing by Section Length

1. **Section <350 Words:**  
   - **Action:**  
     Use the section as a single chunk without further splitting.
     
2. **Section Between 350 and 750 Words:**  
   - **Action:**  
     Divide the section into chunks if needed (each ≤350 words) and then apply the extraction process so that the section is reduced to about 350 words.
     
3. **Section >750 Words:**
     Subdivide the section into multiple subsections using the same logic:
     - First, split the section into chunks where each chunk is between 350 and 750 words.
     - Then, for each chunk, if its length is between 350 and 750 words, further divide it into smaller chunks (each ≤350 words) and extract until the entire subsection is reduced to about 350 words.
     - All subsections should now be of max. 350 words, and each be a chunk.

### B. First Fixed-Ratio Extraction per Chunk

If a section has to be reduced to 350 words and it is divided into chunks than then have to be reduced , we apply an extraction step. Although we will later refine these percentages, a baseline extraction might be defined as follows (these values will be adjusted by a computed compression factor later):
- **Example Baseline Percentages:**  
  - 1st chunk: ~34%
  - 2nd chunk: ~30%
  - 3rd chunk: ~24–25%
  - 4th chunk: ~20%
  - 5th chunk: ~16–17%
After processing each section in this manner, we generate an extractive summary for each section that is approximately 350 words long.

### C. Aggregation

- **Aggregate Sections:**  
  Concatenate all section-level extractive summaries to form the aggregated extraction text, denoted as **Lₑ**.

---

## Step 2: Dependent-Ratio Extraction Refinement

The aggregated extraction **Lₑ** may still be too long. We now further compress it to a target intermediate length **T** (e.g., T ≈ 700 words, within our desired 600–750 word range).

### A. Compute the Compression Factor

1. **Calculate f:**
   
   \[
   f = \frac{T}{Lₑ}
   \]
   
   Here, T is our target length (e.g., 700 words) and Lₑ is the length (in words) of the aggregated extraction.
   
2. **Enforce a Minimum:**  
   If \( f < 0.15 \) (i.e., less than 15%), set \( f = 0.15 \) to avoid over-compression.

### B. Apply Weighted Extraction to Chunks

1. **Divide Lₑ into Chunks:**  
   For the purpose of refinement, treat Lₑ as composed of chunks \( C_1, C_2, \dots, C_n \) (maintaining the order from the original document).

2. **Assign Weights:**  
   Assign a weight \( w_i \) to each chunk, with higher weights for earlier chunks to preserve critical information. For example:
   - \( w_1 = 1.2 \)
   - \( w_2 = 1.0 \)
   - \( w_3 = 0.8 \)
   - \( w_4 = 0.6 \)
   - \( w_5 = 0.5 \)
   - \( w_i = 0.5 \) for \( i > 5 \)
   
3. **Determine Effective Extraction Percentage:**  
   For each chunk \( C_i \) with length \( L_i \), compute its effective extraction percentage \( p_i \) as:
   
   \[
   p_i = \text{clamp}(f \times w_i, \, p_{\min}, \, p_{\max})
   \]
   
   Where:
   - \( p_{\min} \) is the minimum allowed extraction rate (e.g., 15% or 0.15),
   - \( p_{\max} \) is the maximum allowed extraction rate (e.g., 35% or 0.35).
   
4. **Extract Each Chunk:**  
   The number of words extracted from chunk \( C_i \) is:
   
   \[
   E_i = p_i \times L_i
   \]
   
   Sum the \( E_i \) values to form the refined extraction text \( L_{\text{final}} \).

### C. Iterate if Needed

- **Check Length:**  
  If \( L_{\text{final}} \) is still above T (600–750 words), recompute:
  
  \[
  f' = \frac{T}{L_{\text{final}}}
  \]
  
  Clamp \( f' \) to a minimum of 0.15, and reapply the weighted extraction to each chunk. Repeat until \( L_{\text{final}} \) is within the desired range.

---

## Step 3: Final Abstractive Summarization

With \( L_{\text{final}} \) refined to around 600–750 words—sufficient to fit within the context window of a standard BART model—we perform the final abstractive summarization in a single step.

1. **Input:**  
   \( L_{\text{final}} \) (approximately 600–750 words).

2. **Process:**  
   Feed \( L_{\text{final}} \) into an abstractive summarizer (e.g., BART).
   
3. **Decoding Parameters:**  
   Configure BART’s decoding parameters (e.g., min_length and max_length) to generate a final summary in the range of 480–600 words.

---

# Tier 4: Hierarchical Summarization for Documents (20,000–68,000 Words)

In Tier 4, we address longer legal documents by first processing each section individually and then combining these “chunks” into a global extract. Our goal is to produce a global extract that does not exceed 750 words, which is then summarized by BART to yield a final summary of 480–600 words. The extraction percentages are not fixed; they depend on computed compression factors. Below is the detailed process with the associated mathematical logic.

---

## Step 1: Section-Based Pre-Summarization

Each section of the document is processed based on its length:

1. **Section <750 Words:**  
   - **Action:**  
     Apply a baseline BART summarization to the section, producing a summary with a maximum length of 350 words. This 350-word summary becomes the “chunk” for that section.

2. **Section 750–1500 Words:**  
   - **Action:**  
     First, generate an extractive summary of the section (using an extraction process) to reduce it to approximately 600 words. Then, feed this 600-word text into baseline BART to produce a summary of up to 350 words.

3. **Section >1500 Words:**  
   - **Action:**  
     Divide the section into multiple subsections, each with a maximum length of 1500 words. For each subsection, follow the same logic as for sections between 750 and 1500 words (i.e., reduce to about 600 words via extractive summarization, then generate a BART summary capped at 350 words).

After processing, each section (or subsection) is reduced to a “chunk” of at most 350 words. These chunks are then aggregated to form the global extraction, denoted as **Lₑ**.

---

## Step 2: Global Dependent-Ratio Extraction

If the aggregated extraction **Lₑ** exceeds our global target (T ≈ 750 words), we apply an additional extraction step across the chunks. The process is as follows:

### A. Compute the Global Compression Factor

1. **Calculate f:**

   \[
   f = \frac{T}{Lₑ}
   \]
   
   where \( T \) is our target length (e.g., 750 words) and \( Lₑ \) is the total length of the aggregated chunks.

2. **Enforce a Minimum Factor:**  
   If \( f < 0.15 \), then set \( f = 0.15 \) to avoid over-compression.

### B. Apply Weighted Extraction Across Chunks

1. **Assume Lₑ is composed of ordered chunks** \( C_1, C_2, \dots, C_n \) (each up to 350 words).

2. **Assign Weights:**  
   To preserve critical content from earlier chunks, assign weights \( w_i \) that decrease with chunk order, e.g.:
   - \( w_1 = 1.2 \)
   - \( w_2 = 1.0 \)
   - \( w_3 = 0.8 \)
   - \( w_4 = 0.6 \)
   - \( w_5 = 0.5 \)
   - For \( i > 5 \), \( w_i = 0.5 \)

3. **Determine Effective Extraction Percentage:**  
   For each chunk \( C_i \) with length \( L_i \), compute:

   \[
   p_i = \text{clamp}(f \times w_i, \, p_{\min}, \, p_{\max})
   \]
   
   where:
   - \( p_{\min} \) is the minimum extraction rate (e.g., 15% or 0.15),
   - \( p_{\max} \) is the maximum extraction rate (e.g., 35% or 0.35).

4. **Extract from Each Chunk:**  
   The extracted word count from chunk \( C_i \) is:

   \[
   E_i = p_i \times L_i
   \]
   
   The refined global extraction \( L_{\text{final}} \) is obtained by concatenating all \( E_i \). Ideally, the sum \( \sum E_i \) is close to \( T \) (750 words).

### C. Iteration

- If \( L_{\text{final}} \) exceeds the target, recompute a new compression factor:

  \[
  f' = \frac{T}{L_{\text{final}}}
  \]
  
  Clamp \( f' \) to a minimum of 0.15, and reapply the weighted extraction until \( L_{\text{final}} \) is within the desired range (around 750 words).

---

## Step 3: Final Abstractive Summarization

Once the global extraction \( L_{\text{final}} \) is refined to approximately 750 words, it is fed into an abstractive summarizer (baseline BART) to generate the final summary.

1. **Input:**  
   \( L_{\text{final}} \) (≈750 words)

2. **Abstractive Summarization:**  
   Use baseline BART (with a context window suitable for 750 words) to produce a final summary. Set decoding parameters (e.g., min_length and max_length) to target an output of 480–600 words.

---

## Summary of Tier 4 Process

1. **Section-Based Pre-Summarization (Step 1):**
   - Sections <750 words: Summarize to ≤350 words using baseline BART.
   - Sections 750–1500 words: Extract to 600 words, then summarize to ≤350 words.
   - Sections >1500 words: Divide into subsections (max 1500 words each), then for each, extract to 600 words and summarize to ≤350 words.
   - Aggregate all section summaries to form \( Lₑ \).

2. **Global Dependent Extraction (Step 2):**
   - Compute \( f = \frac{750}{Lₑ} \) (clamped to ≥0.15).
   - For each chunk \( C_i \) (with length \( L_i \)), compute \( p_i = \text{clamp}(f \times w_i, \, 0.15, \, 0.35) \).
   - Extract \( E_i = p_i \times L_i \) from each chunk.
   - Concatenate \( E_i \) values; iterate if necessary until the global extraction is ~750 words.

3. **Final Abstractive Summarization (Step 3):**
   - Feed the 750-word refined extraction into BART, configured to produce a final summary of 480–600 words.

This process allows us to handle longer documents (20,000–68,000 words) by first summarizing each section to a manageable chunk, then applying a global dependent extraction (with mathematically defined compression factors) that prioritizes earlier content, and finally generating a concise, coherent final summary with baseline BART.
