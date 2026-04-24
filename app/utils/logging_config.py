"""
Centralized logging configuration for Arabic RAG system.

Provides structured logging setup with file and console handlers,
including log rotation and formatting.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional


class LoggerSetup:
    """Centralized logging configuration."""
    
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DETAILED_FORMAT = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - %(message)s'
    )
    
    @staticmethod
    def setup_logger(
        name: str,
        log_file: Optional[str] = None,
        level: str = "INFO",
        console: bool = True,
        file_size_mb: int = 10,
        backup_count: int = 5,
        detailed: bool = False
    ) -> logging.Logger:
        """
        Setup a logger with file and console handlers.
        
        Args:
            name: Logger name (typically __name__).
            log_file: Path to log file (optional).
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            console: Whether to log to console.
            file_size_mb: Max size of log file before rotation (in MB).
            backup_count: Number of backup log files to keep.
            detailed: Whether to use detailed format with file/line info.
            
        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        formatter = logging.Formatter(
            LoggerSetup.DETAILED_FORMAT if detailed else LoggerSetup.DEFAULT_FORMAT
        )
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=file_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


def setup_system_logging(
    log_dir: str = "./logs",
    level: str = "INFO",
    console: bool = True
) -> None:
    """
    Setup logging for the entire system.
    
    Args:
        log_dir: Directory to store log files.
        level: Logging level.
        console: Whether to log to console.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup root logger
    LoggerSetup.setup_logger(
        "app",
        log_file=os.path.join(log_dir, f"app_{timestamp}.log"),
        level=level,
        console=console,
        detailed=True
    )
    
    # Setup component loggers
    components = [
        "embedding_service",
        "llm_service",
        "rag_pipeline",
        "vector_store",
        "document_loader",
        "cache",
        "errors"
    ]
    
    for component in components:
        LoggerSetup.setup_logger(
            f"app.services.{component}" if "_service" in component or "_store" in component or "_loader" in component or component == component else f"app.utils.{component}",
            log_file=os.path.join(log_dir, f"{component}_{timestamp}.log"),
            level=level,
            console=False
        )
    
    root_logger = logging.getLogger()
    root_logger.info(f"System logging initialized in {log_dir}")


# Application-specific log levels for different modules
LOG_CONFIG = {
    "app": "INFO",
    "app.services.embedding_service": "WARNING",
    "app.services.llm_service": "WARNING",
    "app.services.rag_pipeline": "INFO",
    "app.services.faiss_store": "INFO",
    "app.utils.document_loader": "INFO",
    "app.utils.errors": "INFO",
}


def configure_module_logging() -> None:
    """Configure logging levels for specific modules."""
    for module_name, log_level in LOG_CONFIG.items():
        logger = logging.getLogger(module_name)
        logger.setLevel(getattr(logging, log_level.upper()))
