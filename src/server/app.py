"""
Purple Agent Server for OpenHA

FastAPI server implementing A2A protocol for Purple Agent.
"""

import argparse
import logging
import random
from typing import Optional

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.server.executor import Executor
from src.server.session_manager import SessionManager

from starlette.responses import PlainTextResponse, JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_agent_card(
    host: str,
    port: int,
    card_url: Optional[str] = None,
) -> AgentCard:
    """Create agent card for A2A protocol.
    
    Args:
        host: Server host
        port: Server port
        card_url: Optional custom agent card URL
    
    Returns:
        AgentCard instance
    """
    skill = AgentSkill(
        id="openha-purple-policy",
        name="OpenHA Purple Policy",
        description="Purple policy server for OpenHA hierarchical agent",
        tags=["openha", "minecraft", "vla", "purple", "hierarchical"],
        examples=[],
    )

    agent_card = AgentCard(
        name="OpenHA Purple Agent",
        description="Purple agent wrapper for OpenHA hierarchical VLM agent in Minecraft",
        url=card_url or f"http://{host}:{port}/",
        version="0.1.0",
        default_input_modes=["text", "application/json"],
        default_output_modes=["text", "application/json"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )
    
    return agent_card


def main():
    """Main entry point for Purple Agent server."""
    parser = argparse.ArgumentParser(description="Run OpenHA Purple Agent Server")
    
    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=9019, help="Server port")
    parser.add_argument("--card-url", type=str, help="Agent card URL")
    
    # OpenHA configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default="/workspace/models/CrossAgent",
        help="Path to VLM model"
    )
    parser.add_argument(
        "--grounding-policy-path",
        type=str,
        default="/workspace/models/ROCKET-1.12w_EMA",
        help="Path to grounding policy"
    )
    parser.add_argument(
        "--motion-policy-path",
        type=str,
        default="/workspace/models/Motion-Policy",
        help="Path to motion policy"
    )
    parser.add_argument(
        "--sam-path",
        type=str,
        default="/workspace/woosung/AgentBeats-OpenHA/external/SAM2/checkpoints",
        help="Path to SAM model"
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="eager",
        choices=["eager", "greedy", "grounding", "motion", "text_action"],
        help="OpenHA output mode"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="text_action",
        choices=["motion_coa", "grounding_coa", "coa", "text_action", "grounding", "motion"],
        help="OpenHA output format (for text_action mode)"
    )
    parser.add_argument(
        "--vlm-client-mode",
        type=str,
        default="online",
        choices=["vllm", "hf", "online", "openai", "anthropic"],
        help="VLM client mode"
    )
    parser.add_argument(
        "--model-ips",
        type=str,
        default="localhost",
        help="VLM server URL (for vllm mode)"
    )
    parser.add_argument(
        "--model-ports",
        type=str,
        default="11000",
        help="VLM server URL (for vllm mode)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--state-ttl",
        type=int,
        default=3600,
        help="State TTL in seconds"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    # Additional OpenHA parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--maximum-history-length",
        type=int,
        default=15,
        help="Maximum history length"
    )
    parser.add_argument(
        "--grounding-inference-interval",
        type=int,
        default=4,
        help="Grounding inference interval"
    )
    parser.add_argument(
        "--motion-inference-interval",
        type=int,
        default=4,
        help="Motion inference interval"
    )
    parser.add_argument(
        "--action-chunk-len",
        type=int,
        default=1,
        help="Action chunk length"
    )
    parser.add_argument(
        "--raw-action-type",
        type=str,
        default="text",
        choices=["text", "reserved"],
        help="Raw action type"
    )
    parser.add_argument(
        "--system-message-tag",
        type=str,
        default="text_action",
        help="System message tag for few-shot prompting (optional)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting OpenHA Purple Agent Server")
    logger.info("Host: %s", args.host)
    logger.info("Port: %s", args.port)
    logger.info("Model path: %s", args.model_path)
    logger.info("Output mode: %s", args.output_mode)
    logger.info("VLM client mode: %s", args.vlm_client_mode)
    
    # Create agent card
    agent_card = create_agent_card(
        host=args.host,
        port=args.port,
        card_url=args.card_url,
    )
    
    # Create session manager
    sessions = SessionManager()
    
    # Create executor with shared agent
    model_url = f"http://{random.choice(args.model_ips.split(','))}:{random.choice(args.model_ports.split(','))}/v1"
    executor = Executor(
        sessions=sessions,
        model_path=args.model_path,
        grounding_policy_path=args.grounding_policy_path,
        motion_policy_path=args.motion_policy_path,
        sam_path=args.sam_path,
        output_mode=args.output_mode,
        vlm_client_mode=args.vlm_client_mode,
        model_url=model_url,
        device=args.device,
        state_ttl_seconds=args.state_ttl,
        debug=args.debug,
        temperature=args.temperature,
        maximum_history_length=args.maximum_history_length,
        grounding_inference_interval=args.grounding_inference_interval,
        motion_inference_interval=args.motion_inference_interval,
        action_chunk_len=args.action_chunk_len,
        raw_action_type=args.raw_action_type,
        system_message_tag=args.system_message_tag,
    )
    
    logger.info("Agent card created: %s", agent_card.name)
    logger.info("Executor initialized with shared agent")
    
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    asgi_app = app.build()

    
    @asgi_app.route("/health")
    async def health(request):
        return PlainTextResponse("OK")
    
    @asgi_app.route("/.well-known/agent-card.json")
    async def agent_card_endpoint(request):
        return JSONResponse(agent_card.model_dump())
    
    # Run server
    import uvicorn
    logger.info("Starting server on %s:%s", args.host, args.port)
    uvicorn.run(asgi_app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
