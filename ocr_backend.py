import os
import sys
import logging
import json
from typing import List, Dict, Any, Optional
from label_studio_ml.model import LabelStudioMLBase
import pytesseract
from PIL import Image
import requests
from io import BytesIO
from requests.exceptions import RequestException
from PIL import UnidentifiedImageError

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # This will override any existing logging configuration
)

# Create a separate handler for your class's logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class OCRBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        """Initialize model."""
        super().__init__(**kwargs)
        self.from_name = "transcription"  # from your label config
        self.to_name = "image"           # from your label config
        self.value = "image"             # from your label config

        # Get Label Studio host and API key from environment variables or kwargs
        self.label_studio_host = 'http://localhost:8080'
        self.api_key = '23b0645a285db0c43ecdc29ea32518ed200eca8d'
        
        # Log initialization
        logger.info(f"Initialized with from_name={self.from_name}, to_name={self.to_name}, value={self.value}")

    
    def predict(self, tasks, **kwargs):
        """Generate predictions for OCR"""
        logger.info("Predict called")
        logger.debug(f"Tasks: {tasks}")
        logger.debug(f"Kwargs: {kwargs}")
        
        try:

            task = tasks[-1]

            img_url = task.get('data', {}).get('image')
            if not img_url:
                return {"status": "error", "message": "No image URL found"}

            image = self._get_image_from_url(img_url)
            results = []

            
            annotations = task.get('annotations', {})
            
            annotation = annotations[-1]

            for result in annotation.get('result', []):
                if result.get('type') != 'rectanglelabels':
                    continue

                try:
                    value = result.get('value', {})
                    x = value.get('x', 0) * image.width / 100
                    y = value.get('y', 0) * image.height / 100
                    w = value.get('width', 0) * image.width / 100
                    h = value.get('height', 0) * image.height / 100

                    cropped_image = image.crop((int(x), int(y), int(x + w), int(y + h)))
                    ocr_text = pytesseract.image_to_string(
                        cropped_image,
                        config='--psm 7'
                    ).strip()

                    logger.debug(f"OCR text extracted: {ocr_text}")

                    # Add the text area result with text as array
                    results.append({
                        "original_width": image.width,
                        "original_height": image.height,
                        "image_rotation": 0,
                        "value": {
                            "text": [ocr_text],  # Text as array
                            "x": value.get('x'),
                            "y": value.get('y'),
                            "width": value.get('width'),
                            "height": value.get('height')
                        },
                        "id": result.get('id'),
                        "from_name": "transcription",
                        "to_name": "image",
                        "type": "textarea"
                    })

                    # Add the original rectangle annotation
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing region: {str(e)}")
                    continue

            logger.debug(f"Final results structure:\n" + json.dumps(results, indent=2))
            returnObject = [{
                'result': results,
                'score': 1.0,
                'model_version': 'BBOXOCR'
            }] 
            return returnObject
            # img_url = task.get('data', {}).get('image')
            # if not img_url:
            #     return {"status": "error", "message": "No image URL found"}

            # image = self._get_image_from_url(img_url)
            # results = []

            # # Process each rectangle region
            # for result in annotation.get('result', []):
            #     if result.get('type') != 'rectanglelabels':
            #         continue

            #     try:
            #         value = result.get('value', {})
            #         x = value.get('x', 0) * image.width / 100
            #         y = value.get('y', 0) * image.height / 100
            #         w = value.get('width', 0) * image.width / 100
            #         h = value.get('height', 0) * image.height / 100

            #         cropped_image = image.crop((int(x), int(y), int(x + w), int(y + h)))
            #         ocr_text = pytesseract.image_to_string(
            #             cropped_image,
            #             config='--psm 7'
            #         ).strip()

            #         logger.debug(f"OCR text extracted: {ocr_text}")

            #         # Add the original rectangle annotation
            #         results.append(result)

            #         # Add the text area result with text as array
            #         results.append({
            #             "original_width": image.width,
            #             "original_height": image.height,
            #             "image_rotation": 0,
            #             "value": {
            #                 "text": [ocr_text],  # Text as array
            #                 "x": value.get('x'),
            #                 "y": value.get('y'),
            #                 "width": value.get('width'),
            #                 "height": value.get('height')
            #             },
            #             "id": result.get('id'),
            #             "from_name": "transcription",
            #             "to_name": "image",
            #             "type": "textarea"
            #         })
                    
            #     except Exception as e:
            #         logger.error(f"Error processing region: {str(e)}")
            #         continue

            # logger.debug(f"Final results structure:\n" + json.dumps(results, indent=2))
            # return {
            #     "status": "ok",
            #     "result": results
            # }

        except Exception as e:
            logger.error(f"Error in process_event: {str(e)}")
            logger.exception("Full traceback:")
            return {"status": "error", "message": str(e)}

    def _get_image_from_url(self, url):
        """Fetch the image from the given URL."""
        try:
            if url.startswith('/data/'):
                if not self.label_studio_host:
                    raise ValueError("Label Studio host URL is not configured")
                url = f"{self.label_studio_host.rstrip('/')}{url}"
                
            headers = {'Authorization': f'Token {self.api_key}'} if self.api_key else {}
            
            logger.debug(f"Fetching image from: {url}")
            response = requests.get(url, headers=headers, verify=False, timeout=10)
            response.raise_for_status()
            
            return Image.open(BytesIO(response.content)).convert('RGB')
            
        except Exception as e:
            logger.error(f"Failed to load image from {url}: {str(e)}")
            raise
   
    