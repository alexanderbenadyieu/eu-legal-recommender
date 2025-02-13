# EurLex Scraper Metrics Analysis Report

## Overview
This report provides a detailed analysis of the scraping metrics collected during the EurLex document retrieval process.

## Data Collection Summary
- **Total Metrics Files Collected:** 1,085
- **Metrics Files After Filtering:** 603
- **Filtering Criteria:** Removed sessions with 0 or 1 documents processed

## Detailed Metrics Analysis

### Documents Processed
| Statistic | Value |
|-----------|-------|
| Total Sessions | 603 |
| Mean Documents per Session | 13.64 |
| Median Documents per Session | 8 |
| Minimum Documents | 2 |
| Maximum Documents | 87 |
| Standard Deviation | 15.55 |

**Interpretation:**
- The majority of scraping sessions process between 4-15 documents
- There's significant variability in document processing (high standard deviation)
- Some highly productive sessions processed up to 87 documents

### Requests and Performance
| Metric | Mean | Median | Max |
|--------|------|--------|-----|
| Total Requests | Visible in Plot | - | - |
| Retry Attempts | Visible in Plot | - | - |

### Storage Metrics
| Statistic | Value (Bytes) |
|-----------|---------------|
| Mean Storage Size | 1,398.72 |
| Median Storage Size | 4.63 |
| Maximum Storage Size | 356,735 |
| Standard Deviation | 20,587.32 |

**Observations:**
- Extreme variability in storage size
- Median storage size is quite low (4.63 bytes)
- Some sessions result in significantly larger document storage