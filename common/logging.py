import logging
import copy
import os
import sys
from lightning.pytorch.utilities import rank_zero_only

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Create a new logger
logger = logging.getLogger("Trainer")
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    from lightning_utilities.core.imports import RequirementCache   
    _rich_available = RequirementCache("rich>=10.2.2")
    
    if _rich_available and os.environ.get('SM_HOSTS', None) == None:
        from rich.logging import RichHandler

        class ConditionalRichHandler(RichHandler):
            @rank_zero_only
            def emit(self, record):
                super().emit(record)
            
        handler = ConditionalRichHandler(
            rich_tracebacks=True, 
            show_time=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter("%(levelname)s - %(message)s"))
        
    logger.addHandler(handler)

# Configure logger
loglevel = getattr(logging, "DEBUG", None)
logger.setLevel(loglevel)