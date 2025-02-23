"""
Metrics Collection and Monitoring Module for EUR-Lex Web Scraper

This module provides comprehensive metrics collection and monitoring 
capabilities for the EUR-Lex web scraper. It leverages Prometheus 
client library to track and export various performance and processing metrics.

Key Features:
- Prometheus-based metrics tracking
- Flexible metrics collection and export
- Support for file-based and optional HTTP metrics export
- Detailed tracking of document processing, requests, and performance
- Customizable metrics registry

Metrics Tracked:
- Document processing status and count
- HTTP request statistics
- Retry attempts
- Processing time histograms
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, write_to_textfile
from prometheus_client.core import CollectorRegistry
from loguru import logger


class MetricsCollector:
    """
    A comprehensive metrics collection system for the EUR-Lex web scraper.

    Manages the creation, tracking, and export of performance metrics 
    using Prometheus client library. Provides detailed insights into 
    scraping process, request handling, and document processing.

    Attributes:
        registry (CollectorRegistry): Prometheus metrics registry
        metrics_dir (Path): Directory for storing metrics files
        documents_processed (Counter): Tracks total documents processed
        requests_total (Counter): Tracks total HTTP requests
        retry_attempts (Counter): Tracks retry attempts for requests
        document_processing_time (Histogram): Measures document processing times

    Notes:
        - Supports both file-based and optional HTTP metrics export
        - Automatically creates metrics directory
        - Provides granular metrics with labels for detailed analysis
    """
    
    def __init__(self, metrics_dir: Optional[str] = None, export_port: Optional[int] = None):
        """
        Initialize the metrics collection system.

        Sets up a Prometheus metrics registry and configures various 
        metric collectors for tracking scraper performance.

        Args:
            metrics_dir (Optional[str], optional): 
                Directory path to store metrics files. 
                Defaults to 'metrics' in the current directory.
            export_port (Optional[int], optional): 
                Port number for HTTP metrics export. 
                If None, only file-based export is used.

        Notes:
            - Creates metrics directory if it doesn't exist
            - Initializes counters and histograms for various metrics
            - Supports flexible metrics storage and export configurations
        """
        self.registry = CollectorRegistry()
        self.metrics_dir = Path(metrics_dir) if metrics_dir else Path("metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Document processing metrics
        self.documents_processed = Counter(
            'eurlex_documents_processed_total',
            'Total number of documents processed',
            ['status', 'date'],  # success/failure, processing date
            registry=self.registry
        )
        
        # Request metrics
        self.requests_total = Counter(
            'eurlex_requests_total',
            'Total number of HTTP requests made',
            ['status', 'url'],  # success/failure, URL
            registry=self.registry
        )
        
        self.retry_attempts = Counter(
            'eurlex_retry_attempts_total',
            'Total number of retry attempts',
            ['url'],  # URL that needed retry
            registry=self.registry
        )
        
        # Processing time metrics
        self.document_processing_time = Histogram(
            'eurlex_document_processing_seconds',
            'Time spent processing documents',
            ['doc_id'],  # Document ID
            buckets=(1, 2, 5, 10, 30, 60, 120),
            registry=self.registry
        )
        
        self.journal_processing_time = Histogram(
            'eurlex_journal_processing_seconds',
            'Time spent processing journals',
            ['date'],  # Journal date
            buckets=(10, 30, 60, 120, 300, 600),
            registry=self.registry
        )
        
        # Storage metrics
        self.storage_size = Gauge(
            'eurlex_storage_size_bytes',
            'Total size of stored documents',
            registry=self.registry
        )
        
        # Validation metrics
        self.validation_errors = Counter(
            'eurlex_validation_errors_total',
            'Total number of validation errors',
            ['type', 'details'],  # error type, error details
            registry=self.registry
        )
        
        # Debug metrics
        self.debug_events = Counter(
            'eurlex_debug_events_total',
            'Debug events for troubleshooting',
            ['event_type', 'details'],
            registry=self.registry
        )
    
    def save_metrics(self) -> None:
        """
        Save metrics to local Prometheus text file.

        Notes:
            - Saves metrics in Prometheus text format
            - Provides metrics data for analysis
        """
        try:
            # Save Prometheus format metrics
            prom_file = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prom"
            write_to_textfile(str(prom_file), self.registry)
            
            logger.info(f"Metrics saved to {prom_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
    
    def record_document_processed(self, success: bool = True, date: str = '') -> None:
        """
        Record a processed document.

        Increments the document processing counter with the given status 
        and current date.

        Args:
            success (bool, optional): 
                Processing status. Defaults to True.
            date (str, optional): 
                Processing date. Defaults to ''.

        Notes:
            - Automatically adds current date as a label
            - Provides insights into document processing success rates
        """
        status = 'success' if success else 'failure'
        self.documents_processed.labels(status=status, date=date).inc()
    
    def record_request(self, success: bool = True, url: str = '') -> None:
        """
        Record an HTTP request.

        Tracks the total number of HTTP requests made during scraping, 
        with status and URL labels for detailed analysis.

        Args:
            success (bool, optional): 
                Request status. Defaults to True.
            url (str, optional): 
                The URL of the HTTP request. Defaults to ''.

        Notes:
            - Helps monitor request success rates
            - Provides URL-specific request tracking
        """
        status = 'success' if success else 'failure'
        self.requests_total.labels(status=status, url=url).inc()
    
    def record_retry_attempt(self, url: str = '') -> None:
        """
        Record a retry attempt for a specific URL.

        Tracks the number of retry attempts for failed or problematic requests.

        Args:
            url (str, optional): 
                The URL that required a retry. Defaults to ''.

        Notes:
            - Helps identify potentially unstable or problematic URLs
            - Supports analysis of request reliability
        """
        self.retry_attempts.labels(url=url).inc()
    
    def record_validation_error(self, error_type: str, details: str = '') -> None:
        """
        Record a validation error.

        Tracks the total number of validation errors, with error type 
        and details labels for detailed analysis.

        Args:
            error_type (str): 
                Type of validation error
            details (str, optional): 
                Error details. Defaults to ''.

        Notes:
            - Helps monitor validation error rates
            - Provides error-specific tracking
        """
        self.validation_errors.labels(type=error_type, details=details).inc()
    
    def record_debug_event(self, event_type: str, details: str = '') -> None:
        """
        Record a debug event.

        Tracks debug events for troubleshooting purposes.

        Args:
            event_type (str): 
                Type of debug event
            details (str, optional): 
                Event details. Defaults to ''.

        Notes:
            - Helps with debugging and troubleshooting
            - Provides event-specific tracking
        """
        self.debug_events.labels(event_type=event_type, details=details).inc()
    
    def update_storage_size(self, base_dir: Path) -> None:
        """
        Update total storage size.

        Calculates and updates the total size of stored documents.

        Args:
            base_dir (Path): 
                Base directory for stored documents

        Notes:
            - Helps monitor storage usage
            - Provides insights into storage needs
        """
        total_size = sum(f.stat().st_size for f in base_dir.rglob('*') if f.is_file())
        self.storage_size.set(total_size)
    
    def time_document_processing(self, doc_id: str = '') -> None:
        """
        Context manager for timing document processing.

        Measures and records the time taken to process a document.

        Args:
            doc_id (str, optional): 
                Document ID. Defaults to ''.

        Notes:
            - Provides insights into document processing performance
            - Supports statistical analysis of processing times
        """
        return self.document_processing_time.labels(doc_id=doc_id).time()
    
    def time_journal_processing(self, date: str = '') -> None:
        """
        Context manager for timing journal processing.

        Measures and records the time taken to process a journal.

        Args:
            date (str, optional): 
                Journal date. Defaults to ''.

        Notes:
            - Provides insights into journal processing performance
            - Supports statistical analysis of processing times
        """
        return self.journal_processing_time.labels(date=date).time()
