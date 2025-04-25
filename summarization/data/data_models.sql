-- =======================
-- 1. Processed Documents
-- =======================
CREATE TABLE IF NOT EXISTS processed_documents (
    document_id INTEGER PRIMARY KEY,
    celex_number VARCHAR(50) UNIQUE NOT NULL,
    html_url VARCHAR(2000),
    processed_date TIMESTAMP,
    word_count INTEGER,
    summary TEXT,
    summary_word_count INTEGER,
    compression_ratio FLOAT
);

-- =======================
-- 2. Document Sections
-- =======================
CREATE TABLE IF NOT EXISTS document_sections (
    section_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    title TEXT,
    content TEXT,
    section_type VARCHAR(50),
    section_order INTEGER,
    word_count INTEGER,
    summary TEXT,
    summary_word_count INTEGER,
    compression_ratio FLOAT,
    tier INTEGER,
    FOREIGN KEY (document_id) REFERENCES processed_documents(document_id)
);
