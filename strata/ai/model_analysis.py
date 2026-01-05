"""AI model analysis for basin state interpretation."""
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, List
import logging
import random
import json

import numpy as np

from strata.config import MODELS, MOCK_IBKR_DATA
from strata.db.queries import (
    get_current_basin,
    get_recent_residuals,
    get_basin_position_current,
    write_model_interpretation
)

logger = logging.getLogger(__name__)

# Mock mode for testing without real AI models
MOCK_MODE = True  # TODO: Set to False when integrating real Youtu-LLM


class ModelAnalyzer:
    """
    AI model analyzer for basin state interpretation.

    Queries multiple AI models (router + specialized heads) to interpret
    basin state. Model disagreement is the primary signal for regime uncertainty.
    """

    def __init__(self, model_configs: Optional[Dict] = None, use_mock: bool = MOCK_MODE):
        """
        Initialize model analyzer.

        Args:
            model_configs: Model configuration dict (defaults to config.MODELS)
            use_mock: Use mock responses instead of real models
        """
        self.model_configs = model_configs if model_configs is not None else MODELS
        self.use_mock = use_mock

        # Initialize embedding model (lazy loaded)
        self._embedding_model = None

        if self.use_mock:
            logger.info("ModelAnalyzer initialized in MOCK_MODE")
        else:
            logger.info("ModelAnalyzer initialized for real model queries")
            # TODO: Initialize Youtu-LLM connection

    def trigger_model_analysis(
        self,
        basin_id: str,
        timestamp: datetime,
        asset: str,
        timescale: str
    ) -> int:
        """
        Query all models to interpret basin state.

        This method:
        1. Loads basin geometry and recent residuals for context
        2. Constructs analysis prompt with market state data
        3. Queries router model and all specialized heads
        4. Parses model responses into structured format
        5. Generates semantic embeddings for disagreement calculation
        6. Writes all interpretations to model_interpretation table

        Args:
            basin_id: Basin identifier
            timestamp: Analysis timestamp
            asset: Asset symbol
            timescale: Time resolution

        Returns:
            Number of model interpretations generated
        """
        logger.info(f"Triggering model analysis for {basin_id}")

        # Get basin geometry
        basin = get_current_basin(asset, timescale)
        if not basin:
            logger.warning(f"No basin found for {asset} {timescale}")
            return 0

        # Get recent residuals
        residuals = get_recent_residuals(asset, timescale, n=10)
        if not residuals:
            logger.warning(f"No residuals found for {asset} {timescale}")
            return 0

        # Get position
        position = get_basin_position_current(asset, timescale)

        # Extract context
        context = self._prepare_context(basin, residuals, position)

        # Construct prompt
        prompt = self._construct_prompt(asset, timescale, context)

        logger.debug(f"Analysis prompt:\n{prompt}")

        # Query all models
        interpretations_count = 0

        # Query router
        router_id = self.model_configs['router']
        router_response = self.query_model(router_id, prompt, context)

        if router_response:
            self._write_interpretation(
                router_id, basin_id, timestamp, timescale,
                router_response, prompt
            )
            interpretations_count += 1

        # Query specialized heads
        for head_id in self.model_configs['heads']:
            head_response = self.query_model(head_id, prompt, context)

            if head_response:
                self._write_interpretation(
                    head_id, basin_id, timestamp, timescale,
                    head_response, prompt
                )
                interpretations_count += 1

        logger.info(
            f"Generated {interpretations_count} model interpretations for {basin_id}"
        )

        return interpretations_count

    def query_model(
        self,
        model_id: str,
        prompt: str,
        context: Dict
    ) -> Optional[Dict]:
        """
        Query a Youtu-LLM model.

        For Phase 3 initial implementation, uses MOCK_MODE with realistic
        dummy responses. Mock responses have realistic variance and disagreement.

        Args:
            model_id: Model identifier
            prompt: Analysis prompt
            context: Market state context

        Returns:
            Dict with model response in structured format
        """
        if self.use_mock:
            return self._generate_mock_response(model_id, context)
        else:
            # TODO: Implement real Youtu-LLM API call
            logger.error("Real Youtu-LLM integration not yet implemented")
            return None

    def _generate_mock_response(self, model_id: str, context: Dict) -> Dict:
        """
        Generate realistic mock model response.

        Mock responses:
        - Add random noise to base assessment
        - Different models disagree slightly in stable regimes
        - Different models disagree significantly in unstable regimes

        Args:
            model_id: Model identifier
            context: Market state context

        Returns:
            Mock structured response
        """
        # Seed based on model_id for consistent but different responses
        model_seed = hash(model_id) % 1000
        rng = random.Random(model_seed + int(context['timestamp'].timestamp()))

        actual_center = float(context['center_location'])
        actual_width = float(context['basin_width'])
        current_residual = float(context['current_residual'])
        distance_to_center = abs(current_residual - actual_center)

        # Determine if regime is stable
        # Stable if: distance to center is small, basin width is reasonable
        is_stable = (distance_to_center < actual_width * 0.3) and (actual_width < 5.0)

        # Base stability score
        base_stability = 0.8 if is_stable else 0.35

        # Model-specific bias (some models more optimistic/pessimistic)
        model_bias = {
            'youtu_router': 0.0,       # Neutral
            'head_vol': -0.1,          # Slightly pessimistic
            'head_corr': 0.05,         # Slightly optimistic
            'head_temporal': 0.0,      # Neutral
            'head_stability': 0.1      # Optimistic
        }.get(model_id, 0.0)

        # Add noise - more noise when unstable
        noise_scale = 0.1 if is_stable else 0.25

        # Generate center estimate
        center_estimate = actual_center + rng.gauss(0, noise_scale)

        # Generate boundary estimates
        boundary_upper_estimate = center_estimate + (actual_width / 2) + rng.gauss(0, noise_scale * 2)
        boundary_lower_estimate = center_estimate - (actual_width / 2) + rng.gauss(0, noise_scale * 2)

        # Generate stability score
        stability_score = base_stability + model_bias + rng.gauss(0, noise_scale * 0.5)
        stability_score = max(0.0, min(1.0, stability_score))

        # Confidence (lower when unstable)
        confidence = rng.uniform(0.75, 0.95) if is_stable else rng.uniform(0.5, 0.75)

        # Regime type
        if stability_score > 0.7:
            regime_type = 'stable'
        elif stability_score > 0.4:
            regime_type = 'transitional'
        elif rng.random() > 0.5:
            regime_type = 'bifurcating'
        else:
            regime_type = 'collapsing'

        # Reasoning
        reasoning = (
            f"Mock response from {model_id}. "
            f"Basin appears {'stable' if is_stable else 'unstable'} "
            f"with center at {center_estimate:.3f}. "
            f"Current residual ({current_residual:.3f}) is "
            f"{'within' if distance_to_center < actual_width/2 else 'near edge of'} basin."
        )

        response = {
            'regime_type': regime_type,
            'center_estimate': center_estimate,
            'boundary_upper_estimate': boundary_upper_estimate,
            'boundary_lower_estimate': boundary_lower_estimate,
            'stability_score': stability_score,
            'confidence': confidence,
            'reasoning': reasoning
        }

        logger.debug(
            f"Mock response from {model_id}: regime={regime_type}, "
            f"stability={stability_score:.3f}, confidence={confidence:.3f}"
        )

        return response

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding for text.

        Uses sentence-transformers library with 'all-MiniLM-L6-v2' model
        (384 dimensions, matches database schema).

        Args:
            text: Text to embed

        Returns:
            np.ndarray of shape (384,)
        """
        # Lazy load embedding model
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence embedding model...")
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")

        # Generate embedding
        embedding = self._embedding_model.encode(text)

        return embedding

    def _prepare_context(
        self,
        basin: Dict,
        residuals: List[Dict],
        position: Optional[Dict]
    ) -> Dict:
        """
        Prepare market state context for model prompt.

        Args:
            basin: Basin geometry data
            residuals: Recent residuals
            position: Position data (optional)

        Returns:
            Context dict
        """
        # Extract residual values
        residual_values = [float(r['normalized_residual']) for r in residuals]

        context = {
            'timestamp': basin['timestamp'],
            'center_location': float(basin['center_location']),
            'basin_width': float(basin['basin_width']),
            'boundary_upper': float(basin['boundary_upper']),
            'boundary_lower': float(basin['boundary_lower']),
            'center_velocity': float(basin['center_velocity']),
            'current_residual': residual_values[0] if residual_values else 0.0,
            'recent_residuals': residual_values,
            'residual_mean': float(np.mean(residual_values)),
            'residual_std': float(np.std(residual_values)),
        }

        if position:
            context.update({
                'distance_to_center': float(position['distance_to_center']),
                'distance_to_boundary': float(position['distance_to_boundary']),
                'position_state': position['position_state'],
                'lag_score': float(position['lag_score'])
            })

        return context

    def _construct_prompt(self, asset: str, timescale: str, context: Dict) -> str:
        """
        Construct analysis prompt for models.

        Args:
            asset: Asset symbol
            timescale: Time resolution
            context: Market state context

        Returns:
            Prompt string
        """
        prompt = f"""You are analyzing market regime stability.

Asset: {asset}
Timescale: {timescale}
Current residual: {context['current_residual']:.4f}
Basin center: {context['center_location']:.4f}
Basin width: {context['basin_width']:.4f}
Distance to center: {context.get('distance_to_center', 'N/A')}
Recent residual trend: {[f"{r:.3f}" for r in context['recent_residuals'][:5]]}

Assess:
1. Regime type: stable/transitional/bifurcating/collapsing
2. Estimate basin center location
3. Estimate basin boundaries (upper and lower)
4. Stability score (0-1, higher = more stable)
5. Your confidence in this assessment (0-1)

Provide your interpretation as JSON:
{{
  "regime_type": "stable",
  "center_estimate": 0.05,
  "boundary_upper_estimate": 2.5,
  "boundary_lower_estimate": -2.5,
  "stability_score": 0.85,
  "confidence": 0.9,
  "reasoning": "Basin is well-defined with clear mean reversion..."
}}"""

        return prompt

    def _write_interpretation(
        self,
        model_id: str,
        basin_id: str,
        timestamp: datetime,
        timescale: str,
        response: Dict,
        prompt: str
    ) -> None:
        """
        Write model interpretation to database.

        Args:
            model_id: Model identifier
            basin_id: Basin identifier
            timestamp: Analysis timestamp
            timescale: Time resolution
            response: Model response dict
            prompt: Original prompt (for embedding generation)
        """
        # Generate embedding from reasoning text
        embedding_text = f"{response['regime_type']}: {response['reasoning']}"
        embedding = self.generate_embedding(embedding_text)

        # Convert embedding to list for pgvector
        embedding_list = embedding.tolist()

        # Prepare data
        data = {
            'model_id': model_id,
            'basin_id': basin_id,
            'timestamp': timestamp,
            'timescale': timescale,
            'regime_type': response['regime_type'],
            'center_estimate': Decimal(str(round(response['center_estimate'], 6))),
            'boundary_upper_estimate': Decimal(str(round(response['boundary_upper_estimate'], 4))),
            'boundary_lower_estimate': Decimal(str(round(response['boundary_lower_estimate'], 4))),
            'stability_score': Decimal(str(round(response['stability_score'], 3))),
            'confidence': Decimal(str(round(response['confidence'], 3))),
            'interpretation_text': response['reasoning'],
            'embedding': embedding_list
        }

        # Write to database
        write_model_interpretation(data)

        logger.debug(
            f"Wrote interpretation from {model_id}: regime={response['regime_type']}, "
            f"stability={response['stability_score']:.3f}"
        )
