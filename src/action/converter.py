"""
OpenHA Action Converter

Converts OpenHA env_action format to Purple Agent agent_action format
using minestudio's built-in conversion pipeline.
"""

import logging
import numpy as np
from typing import Dict, Any

from minestudio.utils.vpt_lib.actions import ActionTransformer
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping

logger = logging.getLogger(__name__)


def noop_action() -> Dict[str, Any]:
    """Return noop action in Purple Agent format.
    
    Returns:
        Dict with buttons=[0] and camera=[60] (center of 11x11 grid)
    """
    return {
        "buttons": [0],
        "camera": [60],  # center (11x11 grid)
    }


class ActionConverter:
    """
    Convert OpenHA env_action to Purple Agent agent_action format.
    
    Uses minestudio's built-in conversion pipeline:
        env_action → policy_action (factored) → agent_action (joint)
    
    OpenHA outputs:
        - 20 individual button fields (0/1)
        - camera as [pitch, yaw] in degrees
    
    Purple Agent requires:
        - buttons as single index (0-2303)
        - camera as single bin index (0-120)
    """
    
    def __init__(
        self,
        camera_binsize: int = 2,
        camera_maxval: int = 10,
        camera_mu: float = 10.0,
        camera_quantization_scheme: str = "mu_law",
    ):
        """
        Initialize converter with minestudio components.
        
        Args:
            camera_maxval: Maximum camera angle value
            camera_binsize: Camera bin size
            camera_mu: Camera mu parameter for quantization
            camera_quantization_scheme: Quantization scheme ('mu_law', 'linear', etc.)
        """
        # Initialize ActionTransformer (env_action ↔ policy_action)
        self.action_transformer = ActionTransformer(
            camera_binsize=camera_binsize,
            camera_maxval=camera_maxval,
            camera_mu=camera_mu,
            camera_quantization_scheme=camera_quantization_scheme,
        )
        
        # Initialize CameraHierarchicalMapping (policy_action ↔ agent_action)
        # Purple Agent uses 11x11 camera grid = 121 bins
        n_camera_bins = 2 * camera_maxval // camera_binsize + 1  
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=n_camera_bins)
        
        logger.info("ActionConverter initialized with n_camera_bins=%d", n_camera_bins)
    
    def env_to_agent(self, env_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenHA env_action to Purple Agent agent_action.
        
        Pipeline:
            1. env_action → policy_action (via ActionTransformer.env2policy)
            2. policy_action → agent_action (via CameraHierarchicalMapping.from_factored)
        
        Args:
            env_action: OpenHA format
                {
                    'forward': 0/1, 'back': 0/1, ..., 
                    'camera': [pitch, yaw]
                }
        
        Returns:
            Purple Agent format
                {
                    "buttons": [button_index],
                    "camera": [camera_bin]
                }
        """
        if env_action is None:
            logger.warning("None action provided, using noop")
            return noop_action()
        
        if not isinstance(env_action, dict):
            logger.error("Invalid action type: %s, using noop", type(env_action))
            return noop_action()
        
        action = {}
        for key, value in env_action.items():
            if key != "camera":
                if isinstance(value, (list, np.ndarray)):
                    action[key] = value
                else:
                    action[key] = [value]
            else:
                if value.ndim != 2:
                    action[key] = np.array(value).reshape(1, -1)
                else:
                    action[key] = value
        
        try:
            # Step 1: env_action → policy_action (factored format)
            # Adds batch dimension and converts to numpy
            action = self.action_transformer.env2policy(action)
            action = self.action_mapper.from_factored(action)
            
            result = {
                "buttons": [int(action["buttons"][0, 0])],
                "camera": [int(action["camera"][0, 0])],
            }
            
            return result
        except Exception as e:
            logger.exception("Failed to convert action: %s", e)
            return noop_action()
    
    @staticmethod
    def get_camera_center() -> int:
        """Get center camera bin for Purple Agent (11x11 grid).
        
        Returns:
            60 (center of 11x11 grid)
        """
        return 60  # (11 * 11) // 2


# Singleton instance for easy import
_converter = None


def get_converter() -> ActionConverter:
    """Get or create singleton ActionConverter instance.
    
    Returns:
        Shared ActionConverter instance with camera_maxval=10 (OpenHA standard)
    """
    global _converter
    if _converter is None:
        # This ensures proper conversion from OpenHA env_action to Purple agent_action
        _converter = ActionConverter(camera_maxval=10)
    return _converter
