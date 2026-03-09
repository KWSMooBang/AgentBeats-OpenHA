"""
OpenHA Purple Agent

Purple Agent wrapper for OpenHA with action conversion.
"""

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import torch

from src.action.converter import noop_action

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Agent state for OpenHA session.
    
    Attributes:
        memory: Policy-specific memory (e.g., hidden states)
        first: Whether this is the first step
        idle_count: Number of idle actions
        task_text: Task description
        task_name: Task name (e.g., "kill_entity:sheep")
    """
    memory: Optional[Any] = None
    first: bool = False
    idle_count: int = 0
    task_text: Optional[str] = None


class OpenHAPurpleAgent:
    """
    Purple Agent Wrapper for OpenHA (Simple Wrapper with Action Conversion)
    
    Wraps OpenHA's interface and converts env_action to agent_action format.
    """
    
    def __init__(
        self,
        # OpenHA initialization parameters
        model_path: str,
        grounding_policy_path: str,
        motion_policy_path: str,
        sam_path: str,
        output_mode: str = "eager",
        raw_action_type: str = "text",
        vlm_client_mode: str = "vllm",
        model_url: str = None,
        device: str = "cuda",
        maximum_history_length: int = 15,
        grounding_inference_interval: int = 4,
        motion_inference_interval: int = 4,
        action_chunk_len: int = 1,
        temperature: float = 0.8,
        **kwargs
    ):
        """Initialize OpenHA Purple Agent.
        
        Args:
            model_path: Path to VLM model
            grounding_policy_path: Path to grounding policy
            motion_policy_path: Path to motion policy
            sam_path: Path to SAM model
            output_mode: Output mode ("eager", "greedy", "grounding", "motion", "text_action")
            raw_action_type: Raw action type ("text" or "reserved")
            vlm_client_mode: VLM client mode ("vllm", "hf", "online", etc.)
            model_url: VLM server URL (for vllm mode)
            device: Device to use ("cuda" or "cpu")
            maximum_history_length: Maximum history length
            grounding_inference_interval: Grounding inference interval
            motion_inference_interval: Motion inference interval
            action_chunk_len: Action chunk length
            temperature: Sampling temperature
            **kwargs: Additional OpenHA parameters
        """
        self._device = torch.device(device)
        self.output_mode = output_mode
        
        logger.info("Initializing OpenHA agent...")
        
        # Import here to avoid circular dependencies
        from openagents.agents.openha import OpenHA
        
        # Initialize OpenHA instance
        self.agent = OpenHA(
            output_mode=output_mode,
            output_format=None,  
            raw_action_type=raw_action_type,
            vlm_client_mode=vlm_client_mode,
            model_path=model_path,
            grounding_policy_path=grounding_policy_path,
            motion_policy_path=motion_policy_path,
            sam_path=sam_path,
            segment_type="Explore",  # Default segment type
            model_url=model_url,
            maximum_history_length=maximum_history_length,
            action_chunk_len=action_chunk_len,
            temperature=temperature,
            grounding_inference_interval=grounding_inference_interval,
            motion_inference_interval=motion_inference_interval,
            **kwargs
        )
        
        # Initialize action converter
        from src.action.converter import get_converter
        self.action_converter = get_converter()
        
        logger.info("OpenHA Purple Agent initialized successfully")
    
    @property
    def device(self) -> torch.device:
        """Get device property.
        
        Returns:
            torch.device instance
        """
        return self._device
    
    def reset(self, task_text: Optional[str] = None, task_name: Optional[str] = None) -> None:
        """Reset agent state.
        
        Args:
            task_text: Task description (e.g., "Kill 3 sheep")
            task_name: Task name (e.g., "kill_entity:sheep", optional)
        """
        if task_text:
            logger.info("Resetting OpenHA with instruction=%s", task_text)
            self.agent.reset(
                instruction=task_text,
                task_name=task_text
            )
        else:
            logger.debug("OpenHAPurpleAgent reset called without task")
    
    def initial_state(self, task_text: Optional[str] = None) -> AgentState:
        """Create initial agent state.
        
        Args:
            task_text: Task description (e.g., "Kill 3 sheep")
        
        Returns:
            Initial AgentState
        """
        return AgentState(
            memory=None,
            first=True,
            idle_count=0,
            task_text=task_text
        )
    
    def act(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], AgentState]:
        """Generate action from observation.
        
        Pipeline:
            1. Extract image from observation
            2. Get env_action from OpenHA
            3. Convert env_action to agent_action
            4. Update state
        
        Args:
            obs: Observation dict with "image" key (np.ndarray[H, W, 3])
            state: Current agent state
            deterministic: Whether to use deterministic sampling (unused)
        
        Returns:
            Tuple of (agent_action, new_state)
            agent_action can be None if error occurs
        """
        try:
            # 1. Extract image from observation
            image = obs.get("image")
            if image is None:
                logger.warning("No image in observation")
                state.first = False
                return None, state
            
            # 2. Get env_action from OpenHA
            info = { "pov" : image }
            
            env_action = self.agent.get_action(
                obs=image,
                info=info,
                verbose=False
            )
            
            logger.info("OpenHA returned env_action: %s", env_action)
            
            # 3. Convert env_action to agent_action
            if env_action is None or not isinstance(env_action, dict):
                logger.warning("OpenHA returned invalid action (None or not dict), using noop")
                agent_action = noop_action()
            else:
                try:
                    agent_action = self.action_converter.env_to_agent(env_action)
                    logger.info("Converted env_action to agent_action: %s", agent_action)
                except Exception as conv_error:
                    logger.exception("Failed to convert env_action to agent_action: %s", conv_error)
                    agent_action = noop_action()
            
            # Validate agent_action has required keys
            if not isinstance(agent_action, dict) or "buttons" not in agent_action or "camera" not in agent_action:
                logger.warning("Invalid agent_action format, using noop")
                agent_action = noop_action()
            
            # 4. Update state
            new_state = AgentState(
                memory=state.memory,
                first=False,
                idle_count=state.idle_count,
                task_text=state.task_text
            )
    
            return agent_action, new_state
            
        except Exception as e:
            logger.exception("OpenHAPurpleAgent.act failed: %s", e)
            state.first = False
            # Return noop_action instead of None
            return noop_action(), state
    