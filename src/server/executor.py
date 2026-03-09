"""
Purple Agent Executor for A2A Protocol

Handles init and observation messages from Green Agent.
"""

import json
import logging
import time
import base64
import numpy as np
from typing import Dict, Any, Optional
from io import BytesIO
from PIL import Image

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, TextPart
from a2a.utils import new_agent_text_message

from src.protocol.models import InitPayload, ObservationPayload, ActionPayload, AckPayload
from src.agent.agent import OpenHAPurpleAgent, AgentState
from src.action.converter import noop_action
from src.server.session_manager import SessionManager

logger = logging.getLogger(__name__)


class Executor(AgentExecutor):
    """
    Purple Policy Executor (A2A-compatible)
    
    Contract:
      - Input: JSON message (init/obs)
      - Output: ONE terminal TaskStatusUpdateEvent via TaskUpdater.complete()
      - No streaming, no multi-event lifecycle
      - Never returns None
    
    Design:
      - Single agent instance shared across all contexts
      - Per-context state management
    """

    def __init__(
        self,
        sessions: SessionManager,
        *,
        model_path: str,
        grounding_policy_path: str,
        motion_policy_path: str,
        sam_path: str,
        output_mode: str = "eager",
        vlm_client_mode: str = "vllm",
        model_url: Optional[str] = None,
        device: Optional[str] = None,
        state_ttl_seconds: Optional[int] = 60 * 60,
        debug: bool = False,
        **kwargs
    ):
        """Initialize executor with shared agent.
        
        Args:
            sessions: SessionManager instance
            model_path: Path to VLM model
            grounding_policy_path: Path to grounding policy
            motion_policy_path: Path to motion policy
            sam_path: Path to SAM model
            output_mode: Output mode for OpenHA
            vlm_client_mode: VLM client mode
            model_url: VLM server URL
            device: Device to use
            state_ttl_seconds: State TTL in seconds
            debug: Enable debug mode
            **kwargs: Additional OpenHA parameters
        """
        self.sessions = sessions
        self.model_path = model_path
        self.grounding_policy_path = grounding_policy_path
        self.motion_policy_path = motion_policy_path
        self.sam_path = sam_path
        self.output_mode = output_mode
        self.vlm_client_mode = vlm_client_mode
        self.model_url = model_url
        self._state_ttl_seconds = state_ttl_seconds
        self._debug = bool(debug)
        self._device = device
        self.kwargs = kwargs

        # Create single shared agent (all contexts use this)
        logger.info("Initializing agent in Executor.__init__...")
        self.agent = OpenHAPurpleAgent(
            model_path=self.model_path,
            grounding_policy_path=self.grounding_policy_path,
            motion_policy_path=self.motion_policy_path,
            sam_path=self.sam_path,
            output_mode=self.output_mode,
            vlm_client_mode=self.vlm_client_mode,
            model_url=self.model_url,
            device=self._device or "cuda",
            **self.kwargs,
        )
        logger.info("Agent initialized successfully")

        # Context-specific state management
        self.agent_states: Dict[str, AgentState] = {}
        self._agent_state_touched_at: Dict[str, float] = {}
        self._last_actions: Dict[str, Dict[str, Any]] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue=None
    ) -> Message:
        """Main entrypoint for A2A protocol.
        
        Args:
            context: Request context (session_id, context_id, task_id, message)
            event_queue: Event queue for task updates
        
        Returns:
            Response message
        """
        # Extract context info
        msg = getattr(context, "message", None)
        context_id = (
            getattr(msg, "context_id", None) 
            or getattr(context, "context_id", None) 
            or "default"
        )
        task_id = (
            getattr(msg, "task_id", None) 
            or getattr(context, "task_id", None) 
            or context_id
        )
        
        # Create task updater
        task_updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )
        
        try:
            # 1. Extract message from context
            message = context.message
            
            # 2. Extract text from message
            text = self._extract_text(message)
            if not text:
                return await self._error_response(task_updater, "No text in message")
            
            # 3. Parse JSON payload
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return await self._error_response(task_updater, "Invalid JSON")
            
            # 4. Route by message type
            msg_type = payload.get("type")
            
            if msg_type == "init":
                return await self._handle_init(payload, context_id, task_id, task_updater)
            elif msg_type == "obs":
                return await self._handle_obs(payload, context_id, task_id, task_updater)
            else:
                return await self._error_response(task_updater, f"Unknown message type: {msg_type}")
        
        except Exception as e:
            logger.exception("Executor.execute failed: %s", e)
            noop_payload = noop_action()
            try:
                response = new_agent_text_message(json.dumps(noop_payload))
                await task_updater.complete(message=response)
                return response
            except Exception as e2:
                logger.exception("Failed to send noop response: %s", e2)
                return new_agent_text_message(json.dumps(noop_payload))

    async def _handle_init(
        self,
        payload: Dict[str, Any],
        context_id: str,
        task_id: str,
        task_updater: TaskUpdater
    ) -> Message:
        """Handle init message.
        
        Process:
            1. Parse InitPayload
            2. Reset agent with task description
            3. Create initial state
            4. Return ack
        
        Args:
            payload: Init payload dict
            context_id: Context ID
            task_id: Task ID
            task_updater: Task updater for sending response
        
        Returns:
            Ack message
        """
        try:
            init_payload = InitPayload(**payload)
        except Exception as e:
            logger.error("Invalid init payload: %s", e)
            return await self._error_response(task_updater, f"Invalid init payload: {e}")
        
        task_text = init_payload.text
        logger.info("INIT context=%s task=%s", context_id, task_text)
        
        # Reset agent and create initial state
        self.agent.reset(task_text=task_text)
        state = self.agent.initial_state(task_text=task_text)
        self.agent_states[context_id] = state
        self._touch(context_id)
        
        # Return ack
        ack_payload = AckPayload(success=True, message="Initialized")
        response = new_agent_text_message(ack_payload.model_dump_json())
        await task_updater.complete(message=response)
        return response

    async def _handle_obs(
        self,
        payload: Dict[str, Any],
        context_id: str,
        task_id: str,
        task_updater: TaskUpdater
    ) -> Message:
        """Handle observation message.
        
        Process:
            1. Parse ObservationPayload
            2. Decode image
            3. Generate action from agent
            4. Convert to ActionPayload
            5. Return action
        
        Args:
            payload: Observation payload dict
            context_id: Context ID
            task_id: Task ID
            task_updater: Task updater for sending response
        
        Returns:
            Action message
        """
        try:
            obs_payload = ObservationPayload(**payload)
        except Exception as e:
            logger.error("Invalid obs payload: %s", e)
            return await self._error_response(task_updater, f"Invalid obs payload: {e}")
        
        step = obs_payload.step
        obs_str = obs_payload.obs
        
        # Decode image
        try:
            image = self._decode_image(obs_str)
        except Exception as e:
            logger.error("Failed to decode image: %s", e)
            return await self._error_response(task_updater, f"Failed to decode image: {e}")
        
        obs_dict = { "image": image }
        
        # Get state
        state = self.agent_states.get(context_id)
        
        if state is None:
            logger.warning("No state for context=%s, returning noop", context_id)
            noop_payload = ActionPayload(
                action_type="agent",
                buttons=noop_action()["buttons"],
                camera=noop_action()["camera"],
            )
            response = new_agent_text_message(noop_payload.model_dump_json())
            await task_updater.complete(message=response)
            return response
        
        # Generate action (shared agent, context-specific state)
        action, new_state = self.agent.act(obs=obs_dict, state=state)
        
        # Update state
        self.agent_states[context_id] = new_state
        self._touch(context_id)
        self._last_actions[context_id] = action
        
        # Convert to ActionPayload
        # Check if action is valid (not None and has required keys)
        if action is None or not isinstance(action, dict) or "buttons" not in action or "camera" not in action:
            logger.warning("Invalid action from agent (step=%d), using noop", step)
            action = noop_action()
        
        action_payload = ActionPayload(
            action_type="agent",
            buttons=action["buttons"],
            camera=action["camera"],
        )
        
        response = new_agent_text_message(action_payload.model_dump_json())
        await task_updater.complete(message=response)
        return response

    def _extract_text(self, msg: Optional[Message]) -> Optional[str]:
        """Extract text from A2A Message.
        
        Args:
            msg: A2A Message
        
        Returns:
            Extracted text or None
        """
        if msg is None:
            return None
        parts = getattr(msg, "parts", None)
        if not isinstance(parts, list):
            return None
        for part in parts:
            # Try multiple access patterns for A2A Message structure
            root = getattr(part, "root", None)
            if isinstance(root, TextPart) and isinstance(root.text, str):
                return root.text
            if isinstance(part, TextPart) and isinstance(part.text, str):
                return part.text
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    return text
        return None

    def _decode_image(self, obs_str: str) -> np.ndarray:
        """Decode base64 encoded image.
        
        Args:
            obs_str: Base64 encoded image string
        
        Returns:
            Numpy array of shape [H, W, 3]
        """
        img_bytes = base64.b64decode(obs_str)
        img = Image.open(BytesIO(img_bytes))
        img_array = np.array(img)
        return img_array

    def _touch(self, context_id: str):
        """Update last touched time for context.
        
        Args:
            context_id: Context ID to touch
        """
        self._agent_state_touched_at[context_id] = time.time()

    async def _error_response(self, task_updater: TaskUpdater, error_msg: str) -> Message:
        """Create error response.
        
        Args:
            task_updater: Task updater
            error_msg: Error message
        
        Returns:
            Error ack message
        """
        ack_payload = AckPayload(success=False, message=error_msg)
        response = new_agent_text_message(ack_payload.model_dump_json())
        await task_updater.complete(message=response)
        return response

    async def cancel(self, context: RequestContext, event_queue=None) -> None:
        """Cancel execution for a context.
        
        Args:
            context: Request context to cancel
            event_queue: Event queue (unused)
        """
        context_id = context.context_id
        logger.warning(f"Canceling execution for context {context_id}")
        
        # Clean up context state
        if context_id in self.agent_states:
            del self.agent_states[context_id]
        if context_id in self._agent_state_touched_at:
            del self._agent_state_touched_at[context_id]
        if context_id in self._last_actions:
            del self._last_actions[context_id]
