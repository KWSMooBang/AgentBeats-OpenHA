"""
Protocol Models for Purple Agent A2A Communication

Defines message formats for init, observation, action, and ack payloads.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class InitPayload(BaseModel):
    """Initialization message from Green Agent to Purple Agent.
    
    Attributes:
        type: Message type (always "init")
        text: Task description (e.g., "Kill 3 sheep")
    """
    type: str = Field(default="init", description="Message type")
    text: str = Field(..., description="Task description")


class ObservationPayload(BaseModel):
    """Observation message from Green Agent to Purple Agent.
    
    Attributes:
        type: Message type (always "obs")
        step: Current step number
        obs: Base64-encoded image observation
    """
    type: str = Field(default="obs", description="Message type")
    step: int = Field(..., description="Current step number")
    obs: str = Field(..., description="Base64-encoded image")


class ActionPayload(BaseModel):
    """Action message from Purple Agent to Green Agent.
    
    Attributes:
        type: Message type (always "action")
        action_type: Action format ("agent" or "env")
        buttons: Button action (list of indices)
        camera: Camera action (list of indices)
    """
    type: str = Field(default="action", description="Message type")
    action_type: str = Field(default="agent", description="Action format")
    buttons: List[int] = Field(..., description="Button action indices")
    camera: List[int] = Field(..., description="Camera action indices")


class AckPayload(BaseModel):
    """Acknowledgment message for init requests.
    
    Attributes:
        type: Message type (always "ack")
        success: Whether initialization succeeded
        message: Optional status message
    """
    type: str = Field(default="ack", description="Message type")
    success: bool = Field(..., description="Success status")
    message: Optional[str] = Field(default="", description="Status message")


class AgentCard(BaseModel):
    """Agent card metadata for A2A protocol.
    
    Attributes:
        name: Agent name
        description: Agent description
        url: Agent server URL
        version: Agent version
        default_input_modes: Supported input modes
        default_output_modes: Supported output modes
        capabilities: Agent capabilities
        skills: List of agent skills
    """
    name: str
    description: str
    url: str
    version: str
    default_input_modes: List[str]
    default_output_modes: List[str]
    capabilities: dict
    skills: List[dict]


class AgentSkill(BaseModel):
    """Agent skill definition.
    
    Attributes:
        id: Skill ID
        name: Skill name
        description: Skill description
        tags: Skill tags
        examples: Example usages
    """
    id: str
    name: str
    description: str
    tags: List[str]
    examples: List[str]


class AgentCapabilities(BaseModel):
    """Agent capabilities definition.
    
    Attributes:
        streaming: Whether agent supports streaming
    """
    streaming: bool = False
