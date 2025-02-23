-- =======================
-- 1. documents
-- =======================
CREATE TABLE documents (
    document_id             SERIAL PRIMARY KEY,          
    celex_number            VARCHAR(50) UNIQUE NOT NULL, -- El celex es el identificador que siempre debe aparecer 
    title                   TEXT NOT NULL,
    identifier              VARCHAR(100), -- Este identificador puede ser NULL para algunos tipos de iniciativa
    eli_uri                 VARCHAR(2000), -- URL oficial de la base de datos de EURLEX
    html_url                VARCHAR(2000), -- URL del documento en formato HTML (pensado para adjuntar a informe)
    pdf_url                 VARCHAR(2000), -- URL del documento en formato PDF (pensado para adjuntar a informe)
    responsible_body_id     INTEGER, 
    form_id                 INTEGER, 
    date_of_document        DATE, -- Fecha en la que se tramitó el documento
    date_of_effect          DATE, -- Empieza a ser vinculante
    date_of_end_validity    DATE, -- Deja de ser vinculante
    content                 TEXT -- Texto de la iniciativa

    FOREIGN KEY (responsible_body_id)
        REFERENCES responsible_bodies(body_id)
        ON DELETE SET NULL,

    FOREIGN KEY (form_id)
        REFERENCES forms(form_id)
        ON DELETE SET NULL
);

-- =======================
-- 1.1 look-up tables form and responsible body
-- =======================

CREATE TABLE forms (
    form_id     SERIAL PRIMARY KEY,
    form_name   VARCHAR(100) NOT NULL -- Single-valued,, Es el tipo de iniciativa: Regulation, Directive, etc.
);

CREATE TABLE responsible_bodies (
    body_id     SERIAL PRIMARY KEY, 
    body_name   VARCHAR(200) NOT NULL -- Single-valued, El directorado general responsable de esa regulación, es distinto al autor
);

-- =======================
-- 2. authors
-- =======================
CREATE TABLE authors (
    author_id   SERIAL PRIMARY KEY,
    name        VARCHAR(300) NOT NULL -- Aquí suele aparecer el directorado general + European Parliament / Commission / Council
);

-- ================================
-- 2.1 document_authors (Many-to-many junction table)
-- ================================
CREATE TABLE document_authors (
    document_id INTEGER NOT NULL,
    author_id   INTEGER NOT NULL,
    PRIMARY KEY (document_id, author_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
    FOREIGN KEY (author_id)   REFERENCES authors(author_id)     ON DELETE CASCADE
);

-- ================================
-- 3. eurovoc_descriptors
-- ================================
CREATE TABLE eurovoc_descriptors (
    descriptor_id   SERIAL PRIMARY KEY,
    descriptor_name VARCHAR(300) NOT NULL -- los eurovoc son clasificaciones temáticas oficiales del tesauro de la UE, suelen ser muchas
);
-- https://op.europa.eu/en/web/eu-vocabularies/dataset/-/resource?uri=http://publications.europa.eu/resource/dataset/eurovoc

-- =======================================
-- 3.1 document_eurovoc_descriptors (Many-to-many)
-- =======================================
CREATE TABLE document_eurovoc_descriptors (
    document_id   INTEGER NOT NULL,
    descriptor_id INTEGER NOT NULL,
    PRIMARY KEY (document_id, descriptor_id),
    FOREIGN KEY (document_id)   REFERENCES documents(document_id)         ON DELETE CASCADE,
    FOREIGN KEY (descriptor_id) REFERENCES eurovoc_descriptors(descriptor_id) ON DELETE CASCADE
);

-- =======================
-- 4. subject_matters
-- =======================
CREATE TABLE subject_matters (
    subject_id      SERIAL PRIMARY KEY,
    subject_name    VARCHAR(300) NOT NULL -- temática a nivel legislativo/jurídico, suelen ser 1-2 
);

-- =================================
-- 4.1 document_subject_matters (Many-to-many)
-- =================================
CREATE TABLE document_subject_matters (
    document_id INTEGER NOT NULL,
    subject_id  INTEGER NOT NULL,
    PRIMARY KEY (document_id, subject_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)      ON DELETE CASCADE,
    FOREIGN KEY (subject_id)  REFERENCES subject_matters(subject_id) ON DELETE CASCADE
);

-- =======================
-- 5. directory_codes
-- =======================
CREATE TABLE directory_codes (
    directory_id       SERIAL PRIMARY KEY,
    directory_code     VARCHAR(100) NOT NULL,
    directory_label    VARCHAR(300) NOT NULL
);

-- ================================= 
-- 5.1 document_directory_codes (Many-to-many)
-- =================================
CREATE TABLE document_directory_codes (
    document_id   INTEGER NOT NULL,
    directory_id  INTEGER NOT NULL,
    PRIMARY KEY (document_id, directory_id),
    FOREIGN KEY (document_id)  REFERENCES documents(document_id)       ON DELETE CASCADE,
    FOREIGN KEY (directory_id) REFERENCES directory_codes(directory_id) ON DELETE CASCADE
);