import os
import pytest
import shutil
import tempfile
from src.document_tracker import DocumentTracker

class TestDocumentTracker:
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_document_tracker_initialization(self, temp_data_dir):
        """Test initialization of DocumentTracker."""
        tracker = DocumentTracker(base_dir=temp_data_dir)
        assert os.path.exists(temp_data_dir)

    def test_is_document_processed(self, temp_data_dir):
        """Test document processing tracking."""
        tracker = DocumentTracker(base_dir=temp_data_dir)
        
        # Test unprocessed document
        assert not tracker.is_document_processed("202302105")
        
        # Mark document as processed
        tracker.mark_document_processed("202302105", {
            "celex_number": "32023D2105",
            "title": "Test Document"
        })
        
        # Verify document is now processed
        assert tracker.is_document_processed("202302105")

    def test_mark_document_processed(self, temp_data_dir):
        """Test marking documents as processed."""
        tracker = DocumentTracker(base_dir=temp_data_dir)
        
        metadata = {
            "celex_number": "32023D2105",
            "title": "Test Document"
        }
        
        tracker.mark_document_processed("202302105", metadata)
        
        # Check if file was created
        processed_file_path = os.path.join(
            temp_data_dir, 
            "2023", 
            "10", 
            "20231005", 
            "202302105.json"
        )
        assert os.path.exists(processed_file_path)
        
        # Verify file contents
        with open(processed_file_path, 'r') as f:
            saved_metadata = json.load(f)
            assert saved_metadata == metadata

    def test_load_existing_documents(self, temp_data_dir):
        """Test loading of existing processed documents."""
        tracker = DocumentTracker(base_dir=temp_data_dir)
        
        # Create some mock processed documents
        os.makedirs(os.path.join(temp_data_dir, "2023", "10", "20231005"), exist_ok=True)
        processed_docs = ["202302105", "202302106", "202302107"]
        
        for doc_id in processed_docs:
            with open(os.path.join(temp_data_dir, "2023", "10", "20231005", f"{doc_id}.json"), 'w') as f:
                json.dump({"celex_number": f"32023D{doc_id[-4:]}"}, f)
        
        # Load existing documents
        tracker._load_existing_documents()
        
        # Verify all documents are marked as processed
        for doc_id in processed_docs:
            assert tracker.is_document_processed(doc_id)

    def test_duplicate_document_prevention(self, temp_data_dir):
        """Test prevention of duplicate document processing."""
        tracker = DocumentTracker(base_dir=temp_data_dir)
        
        metadata = {
            "celex_number": "32023D2105",
            "title": "Test Document"
        }
        
        # First processing attempt
        result1 = tracker.mark_document_processed("202302105", metadata)
        assert result1 is True
        
        # Second processing attempt
        result2 = tracker.mark_document_processed("202302105", metadata)
        assert result2 is False  # Should return False for duplicate
