import os
import sys
import logging
from tree_sitter import Language

# Configure logging
logging.basicConfig(
    filename='docs_generation.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Determine the correct file extension based on the OS
if os.name == 'nt':  # Windows
    LIB_EXT = 'dll'
else:
    LIB_EXT = 'so'

# Define the library path using the determined extension
LIB_PATH = os.path.join('build', f'my-languages.{LIB_EXT}')

try:
    # Ensure the 'build/' directory exists
    os.makedirs('build', exist_ok=True)
    
    # Check if the language library already exists
    if not os.path.exists(LIB_PATH):
        logger.info(f"Building Tree-sitter language library at '{LIB_PATH}'...")
        Language.build_library(
            # Store the library in the 'build/' directory
            LIB_PATH,
            [
                'vendor/tree-sitter-javascript',
                'vendor/tree-sitter-typescript/typescript',
            ]
        )
        logger.info("Tree-sitter language library built successfully.")
    else:
        logger.info(f"Tree-sitter language library already exists at '{LIB_PATH}'. Skipping build.")
except OSError as e:
    logger.error(f"OS error while building Tree-sitter language library: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Failed to build Tree-sitter language library: {e}")
    sys.exit(1)

# Initialize language instances
try:
    JAVASCRIPT_LANGUAGE = Language(LIB_PATH, 'javascript')
    TYPESCRIPT_LANGUAGE = Language(LIB_PATH, 'typescript')
    logger.info("Initialized Tree-sitter language parsers successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Tree-sitter languages: {e}")
    sys.exit(1)
