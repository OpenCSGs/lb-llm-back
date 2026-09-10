import numpy as np
import logging
import os
import sys
import pathlib
from typing import List, Dict, Optional
from urllib.parse import unquote
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image
from .result_converter import convert_masks, convert_ocr_lines
from .template_router import resolve_image_route

logger = logging.getLogger(__name__)

ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)
DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_CONFIG = os.getenv('MODEL_CONFIG', 'configs/sam2.1/sam2.1_hiera_l.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt')
USE_THIRD_PARTY_MODELS = os.getenv(
    'USE_THIRD_PARTY_MODELS', 'false'
).lower() in ('1', 'true', 'yes', 'on')

predictor = None
if USE_THIRD_PARTY_MODELS:
    from .providers.entity_segment import VolcengineEntitySegment
    from .providers.errors import ProviderAPIError
    from .templates.mask_segmentation import DoubaoMaskClassifier
    from .templates.ocr import DoubaoOCR
else:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if DEVICE == 'cuda':
        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # build path to the model checkpoint
    sam2_checkpoint = str(os.path.join(ROOT_DIR, "checkpoints", MODEL_CHECKPOINT))
    sam2_model = build_sam2(MODEL_CONFIG, sam2_checkpoint, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)



class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def _get_image_path(self, image_url, task_id):
        media_root = os.getenv('LABEL_STUDIO_CONTAINER_MEDIA_ROOT')
        if media_root and image_url.startswith('/data/'):
            root = pathlib.Path(media_root).resolve()
            candidate = (root / unquote(image_url.removeprefix('/data/'))).resolve()
            if root in candidate.parents and candidate.is_file():
                return str(candidate)
        return self.get_local_path(
            image_url, task_id=task_id,
            ls_host=getattr(self, '_request_ls_url', None),
            ls_access_token=getattr(self, '_request_ls_token', None),
        )

    def set_image(self, image_url, task_id):
        image_path = self.get_local_path(image_url, task_id=task_id)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        predictor.set_image(image)

    def _sam_predict(self, img_url, point_coords=None, point_labels=None, input_box=None, task=None):
        self.set_image(img_url, task.get('id'))
        point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(input_box, dtype=np.float32) if input_box else None

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        mask = masks[0, :, :].astype(np.uint8)
        prob = float(scores[0])
        # logits = logits[sorted_ind]
        return {
            'masks': [mask],
            'probs': [prob]
        }

    def _volcengine_predict(self, img_url, task, entity_segment):
        image_path = self._get_image_path(img_url, task.get('id'))
        return entity_segment.segment_file(image_path)


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Returns the predicted mask for a smart keypoint that has been placed."""

        credentials = kwargs.pop('credentials', {}) or {}
        self._request_ls_token = kwargs.pop('ls_access_token', None)
        self._request_ls_url = (
            kwargs.pop('ls_url', None)
            or kwargs.pop('label_studio_url', None)
        )
        seed_api_key = credentials.get('seed_api_key') or os.getenv('ARK_API_KEY')

        route = resolve_image_route(self)
        value = route.data_key

        if route.is_ocr:
            if not USE_THIRD_PARTY_MODELS:
                raise ValueError(
                    'OCR template detected in dedicated-model mode; connect the '
                    'original dedicated OCR backend or enable USE_THIRD_PARTY_MODELS'
                )
            image_path = self._get_image_path(tasks[0]['data'][value], tasks[0].get('id'))
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            lines = DoubaoOCR(api_key=seed_api_key).recognize_file(image_path, route.labels)
            results, score = convert_ocr_lines(
                route, lines, image_width, image_height
            )
            logger.info(
                'OCR result returned to Label Studio: regions=%d results=%d types=%s',
                len(lines), len(results),
                {result['type'] for result in results},
            )
            return ModelResponse(predictions=[{
                'result': results,
                'model_version': self.get('model_version'),
                'score': score,
            }])

        if not USE_THIRD_PARTY_MODELS and (not context or not context.get('result')):
            # if there is no context, no interaction has happened yet
            return ModelResponse(predictions=[])

        if context and context.get('result'):
            image_width = context['result'][0]['original_width']
            image_height = context['result'][0]['original_height']
        else:
            image_path = self._get_image_path(tasks[0]['data'][value], tasks[0].get('id'))
            with Image.open(image_path) as image:
                image_width, image_height = image.size

        # collect context information
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None
        for ctx in (context or {}).get('result', []):
            x = ctx['value']['x'] * image_width / 100
            y = ctx['value']['y'] * image_height / 100
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]
            if ctx_type == 'keypointlabels':
                point_labels.append(int(ctx.get('is_positive', 0)))
                point_coords.append([int(x), int(y)])
            elif ctx_type == 'rectanglelabels':
                box_width = ctx['value']['width'] * image_width / 100
                box_height = ctx['value']['height'] * image_height / 100
                input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]

        print(f'Point coords are {point_coords}, point labels are {point_labels}, input box is {input_box}')

        img_url = tasks[0]['data'][value]
        if USE_THIRD_PARTY_MODELS:
            entity_segment = VolcengineEntitySegment(
                access_key=credentials.get('entity_segment_access_key'),
                secret_key=credentials.get('entity_segment_secret_key'),
            )
            mask_classifier = DoubaoMaskClassifier(api_key=seed_api_key)
            predictor_results = self._volcengine_predict(img_url, tasks[0], entity_segment)
            template_labels = route.labels
            if mask_classifier:
                image_path = self._get_image_path(img_url, tasks[0].get('id'))
                try:
                    result_labels = mask_classifier.classify(
                        image_path, predictor_results['masks'], template_labels
                    )
                except ProviderAPIError:
                    raise
                except Exception:
                    fallback_label = template_labels[0]
                    result_labels = [fallback_label] * len(predictor_results['masks'])
                    logger.exception(
                        'Doubao mask classification failed; returning masks with fallback label %s',
                        fallback_label,
                    )
            else:
                selected_label = selected_label or template_labels[0]
                result_labels = [selected_label] * len(predictor_results['masks'])
        else:
            predictor_results = self._sam_predict(
                img_url=img_url,
                point_coords=point_coords or None,
                point_labels=point_labels or None,
                input_box=input_box,
                task=tasks[0]
            )
            result_labels = [selected_label] * len(predictor_results['masks'])

        results, score = convert_masks(
            route=route,
            masks=predictor_results['masks'],
            probabilities=predictor_results['probs'],
            labels=result_labels,
            width=image_width,
            height=image_height,
        )
        predictions = [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': score,
        }]

        return ModelResponse(predictions=predictions)
