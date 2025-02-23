import pytest
import json
from jsonschema import ValidationError
from src.validation import validate_metadata, METADATA_SCHEMA

def test_valid_metadata():
    """Test validation with a complete, valid metadata dictionary."""
    valid_metadata = {
        "celex_number": "32023R2105",
        "title": "Sample Document Title",
        "identifier": "ST/11940/2023/INIT",
        "eli_uri": "http://data.europa.eu/eli/dec/2023/2105",
        "html_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202302105",
        "pdf_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202302105",
        "dates": {
            "Date of document": "07/09/2023",
            "Date of effect": "07/09/2023",
            "Date of end of validity": "No end date"
        },
        "authors": ["Council of the European Union"],
        "responsible_body": "",
        "form": "Decision",
        "eurovoc_descriptors": ["Albania", "agreement (EU)"],
        "subject_matters": ["External relations"],
        "directory_codes": ["11.40.10.40"],
        "directory_descriptions": ["External relations / Bilateral agreements"]
    }
    
    try:
        validate_metadata(valid_metadata)
    except ValidationError:
        pytest.fail("Valid metadata failed validation")

def test_minimal_metadata():
    """Test validation with minimal required metadata."""
    minimal_metadata = {
        "celex_number": "32023R2105",
        "title": "Minimal Document Title"
    }
    
    try:
        validate_metadata(minimal_metadata)
    except ValidationError:
        pytest.fail("Minimal metadata failed validation")

def test_invalid_metadata_missing_title():
    """Test validation fails when title is missing."""
    invalid_metadata = {
        "celex_number": "32023R2105"
    }
    
    with pytest.raises(ValidationError, match="'title' is a required property"):
        validate_metadata(invalid_metadata)

def test_identifier_validation():
    """Test identifier validation logic."""
    valid_identifiers = [
        "ST/11940/2023/INIT",
        "A/2023/1234",
        "B/2024/5678/XYZ"
    ]
    
    invalid_identifiers = [
        "INVALID FORMAT",
        "1234/5678",
        "A/B/C"
    ]
    
    for identifier in valid_identifiers:
        metadata = {
            "celex_number": "32023R2105",
            "title": "Test Document",
            "identifier": identifier
        }
        try:
            validate_metadata(metadata)
        except ValidationError:
            pytest.fail(f"Valid identifier {identifier} failed validation")
    
    for identifier in invalid_identifiers:
        metadata = {
            "celex_number": "32023R2105",
            "title": "Test Document",
            "identifier": identifier
        }
        validate_metadata(metadata)  # Should set identifier to empty string
        assert metadata["identifier"] == ""

def test_celex_number_logging(caplog):
    """Test logging of non-standard CELEX numbers."""
    non_standard_celex = "32023R2147"
    metadata = {
        "celex_number": non_standard_celex,
        "title": "Test Document"
    }
    
    validate_metadata(metadata)
    
    # Check if the CELEX number is logged
    assert f"CELEX number: {non_standard_celex}" in caplog.text
