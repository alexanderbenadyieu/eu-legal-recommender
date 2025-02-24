# EUR-Lex Legal Document Database Model

## Overview
The database is designed to store EU legal documents and their associated metadata in a normalized structure. It supports efficient storage and retrieval of document content, metadata, and relationships between different aspects of the documents.

## Database Entity Relationship Diagram

```
+----------------------+       +------------------------+
|      documents       |       |    responsible_bodies  |
+----------------------+       +------------------------+
| PK: document_id      |       | PK: responsible_body_id|
| celex_number         |<----->| body_name             |
| title                |       +------------------------+
| identifier           |       +------------------------+
| eli_uri              |       |         forms          |
| html_url             |       +------------------------+
| pdf_url              |       | PK: form_id           |
| responsible_body_id   |<----->| form_name             |
| form_id              |       +------------------------+
| date_of_document     |       +------------------------+
| date_of_effect       |       |        authors         |
| date_of_end_validity |       +------------------------+
| content              |       | PK: author_id         |
| html_content         |       | name                  |
+----------------------+       +------------------------+
         |                    |                          ^
         |                    |                          |
         +----------------------------+                  |
                  (via document_authors)              |
```

## Table Descriptions

### Core Tables

#### documents
Primary table storing document information and content
- `document_id`: Primary key
- `celex_number`: Unique EU document identifier
- `title`: Document title
- `identifier`: Document identifier (e.g., C/2023/6433)
- `eli_uri`: European Legislation Identifier URI
- `html_url`: URL to HTML version
- `pdf_url`: URL to PDF version
- `responsible_body_id`: Foreign key to responsible_bodies
- `form_id`: Foreign key to forms
- `date_of_document`: Document creation date
- `date_of_effect`: Date when document takes effect
- `date_of_end_validity`: End of validity date
- `content`: Raw HTML content of the document
- `html_content`: Full HTML content of the document

#### responsible_bodies
Stores EU bodies responsible for documents
- `responsible_body_id`: Primary key
- `body_name`: Name of the responsible body

#### forms
Document form types
- `form_id`: Primary key
- `form_name`: Name of the form (e.g., Regulation, Directive)

#### authors
Document authors
- `author_id`: Primary key
- `name`: Author name

### Junction Tables

#### document_authors
Links documents to their authors
- `document_id`: Foreign key to documents
- `author_id`: Foreign key to authors

#### document_eurovoc_descriptors
Links documents to eurovoc descriptors
- `document_id`: Foreign key to documents
- `descriptor_id`: Foreign key to eurovoc_descriptors

#### document_subject_matters
Links documents to subject matters
- `document_id`: Foreign key to documents
- `subject_id`: Foreign key to subject_matters

#### document_directory_codes
Links documents to directory codes
- `document_id`: Foreign key to documents
- `directory_id`: Foreign key to directory_codes
