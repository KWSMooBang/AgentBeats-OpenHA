"""
MCU Benchmark Rollout

Run MCU benchmark tasks with OpenHA agent.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openagents.agents.openha import OpenHA
from openagents.envs.mcu_env import (
    env_init_mcu,
    run_mcu_task,
    run_mcu_benchmark,
)


def rollout_mcu_benchmark(args):
    """Running MCU benchmark on a specific tasks."""
    print("\n" + "="*70)
    print("ROLLOUT: MCU Benchmark")
    print("="*70 + "\n")
    
    # Initialize agent
    print("Initializing OpenHA agent...")
    agent = OpenHA(
        model_path=args.model_path,
        output_mode=args.output_mode,
        output_format=args.output_format,  # Use same as output_mode
        vlm_client_mode=args.vlm_client_mode,
        model_url=args.model_url,
        model_id=args.model_id,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_message_tag=args.system_message_tag,
        system_message=args.system_message,
        grounding_policy_path=args.grounding_policy_path,
        motion_policy_path=args.motion_policy_path,
        sam_path=args.sam_path,
        grounding_inference_interval=args.grounding_inference_interval,
        motion_inference_interval=args.motion_inference_interval,
        maximum_history_length=args.maximum_history_length,
        raw_action_type=args.raw_action_type,
    )
    
    # Run benchmark
    task_categories = [args.category]
    print(f"Running benchmark on {task_categories} tasks...")
    
    results = run_mcu_benchmark(
        agent=agent,
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir,
        task_categories=task_categories,
        obs_size=(224, 224),
        max_steps=args.max_steps,
        verbose=args.verbose,
    )
    
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"Total Tasks: {results['total_tasks']}")
    print(f"Successful Tasks: {results['successful_tasks']}")
    print(f"Overall Success Rate: {results['overall_success_rate']:.2%}")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout MCU Benchmark")
    
    # Task configuration
    parser.add_argument(
        '--category',
        type=str,
        choices=['building', 'combat', 'crafting', 'decoration', 'ender_dragon', 
                 'explore', 'find', 'mine_diamond_from_scratch', 'mining_and_collecting', 
                 'motion', 'tool_use', 'trapping'],
        default="combat",
        help='Task category to run.'
    )
    parser.add_argument("--tasks_dir", type=str, default="openagents/assets/mcu_tasks", help="Directory containing MCU task YAML files")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")
    parser.add_argument("--max_steps", type=int, default=600, help="Maximum steps per task")
    
    # Agent configuration
    parser.add_argument('--vlm_client_mode', type=str, default="online", choices=["online", "openai", "anthropic", "vllm", "lmdeploy", "hf"])
    parser.add_argument('--output_mode', type=str, default="text_action", choices=["fusion", "eager", "greedy", "grounding", "motion", "text_action"])
    parser.add_argument('--output_format', type=str, default="text_action", choices=["motion_coa", "grounding_coa", "coa", "text_action", "grounding", "motion"])
    parser.add_argument("--system_message_tag", type=str, default="text_action")
    parser.add_argument("--system_message", type=str, default=None)
    parser.add_argument('--raw_action_type', type=str, default="text", choices=["reserved", "text"])
    
    # Model configuration
    parser.add_argument("--model_id", type=str, default="CrossAgent-qwen2vl-7b")
    parser.add_argument("--model_path", type=str, default="/workspace/models/CrossAgent")
    parser.add_argument("--model_url", type=str, default="http://localhost:11000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    
    # Policy paths
    parser.add_argument("--sam_path", type=str, default="facebook/sam2-hiera-base-plus")
    parser.add_argument("--grounding_policy_path", type=str, default="/workspace/models/ROCKET-1.12w_EMA")
    parser.add_argument("--motion_policy_path", type=str, default="/workspace/models/Motion-Policy")
    
    # Inference parameters
    parser.add_argument("--grounding_inference_interval", type=int, default=4)
    parser.add_argument("--motion_inference_interval", type=int, default=4)
    parser.add_argument('--maximum_history_length', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--max_tokens', type=int, default=512)
    
    # Other options
    parser.add_argument('--verbose', type=bool, default=True)
    
    args = parser.parse_args()
    
    rollout_mcu_benchmark(args)
