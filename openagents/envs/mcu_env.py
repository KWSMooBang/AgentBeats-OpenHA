"""
MCU Environment Integration

Provides MinecraftSim environment initialization compatible with MCU task configurations.
"""

import os
import json
import time
import yaml
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from minestudio.simulator import MinecraftSim
from minestudio.simulator.entry import CameraConfig
from minestudio.simulator.callbacks import (
    CommandsCallback, 
    RecordCallback, 
    RewardsCallback,
    JudgeResetCallback,
)


def validate_mcu_config(config: Dict[str, Any]) -> bool:
    """
    Validate MCU task configuration.
    
    Args:
        config: Task configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['text', 'reward_cfg']
    
    for field in required_fields:
        if field not in config:
            print(f"Missing required field: {field}")
            return False
    
    # Validate rewards format
    if not isinstance(config['reward_cfg'], list):
        print("reward_cfg must be a list")
        return False
    
    for reward in config['reward_cfg']:
        if not isinstance(reward, dict):
            print("Each reward must be a dictionary")
            return False
        if 'event' not in reward or 'identity' not in reward:
            print("Each reward must have 'event' and 'identity' fields")
            return False
    
    return True


def list_mcu_tasks(tasks_dir: str, category: Optional[str] = None) -> List[str]:
    """
    List all MCU task YAML files.
    
    Args:
        tasks_dir: Directory containing task YAML files
        category: Optional category filter (e.g., 'combat', 'crafting')
        
    Returns:
        List of YAML file paths
    """
    tasks_dir = Path(tasks_dir)
    
    if category:
        category_dir = tasks_dir / category
        if category_dir.exists():
            return sorted([str(f) for f in category_dir.glob('*.yaml')])
        return []
    
    # List all tasks from all categories
    all_tasks = []
    for category_dir in tasks_dir.iterdir():
        if category_dir.is_dir():
            all_tasks.extend([str(f) for f in category_dir.glob('*.yaml')])
    
    return sorted(all_tasks)


def env_init_mcu(
    yaml_path: str,
    rollout_path: str,
    obs_size: Tuple[int, int] = (224, 224),
    render_size: Tuple[int, int] = (640, 360),
    max_steps: int = 600,
    fps: int = 30,
    camera_cfg: Optional[CameraConfig] = None,
    use_judge_reset: bool = True,
    record_video: bool = True,
    record_raw_action: bool = True,
    **kwargs
) -> Tuple[MinecraftSim, Dict[str, Any]]:
    """
    Initialize MinecraftSim environment using MCU YAML task configuration.
    
    This function creates a MinecraftSim environment compatible with MCU benchmark tasks.
    It parses the YAML file, executes initialization commands, sets up callbacks for
    recording and rewards, and prepares the environment for agent interaction.
    
    Args:
        yaml_path: Path to MCU task YAML file (e.g., "tasks/combat/combat_zombies.yaml")
        rollout_path: Directory path to save recording and results
        obs_size: Observation image size (height, width)
        render_size: Render image size for recording (height, width)
        seed: Random seed for environment
        max_steps: Maximum steps before timeout (for JudgeResetCallback)
        fps: Video recording frame rate
        camera_cfg: Optional camera configuration
        use_judge_reset: Whether to use JudgeResetCallback for timeout
        record_video: Whether to record video (RecordCallback)
        record_raw_action: Whether to create raw_action.jsonl file
        **kwargs: Additional arguments passed to MinecraftSim
        
    Returns:
        (env, task_config): Tuple of initialized MinecraftSim environment and task configuration
        
    Example:
        >>> env, config = env_init_mcu(
        ...     yaml_path="openagents/assets/mcu_tasks/combat/combat_zombies.yaml",
        ...     rollout_path="./output/combat_zombies",
        ...     obs_size=(224, 224),
        ...     max_steps=600,
        ... )
        >>> print(config['text'])
        'Defeat the nearby zombie using weapons and protective gear.'
        >>> obs, info = env.reset()
    """
    # 1. Load and validate YAML configuration
    with open(yaml_path, 'r', encoding='utf-8') as f:
        task_config = yaml.safe_load(f)
    
    if not validate_mcu_config(task_config):
        raise ValueError(f"Invalid MCU task configuration: {yaml_path}")
    
    # 2. Extract configuration (MCU format)
    task_name = Path(yaml_path).stem
    commands = task_config.get('custom_init_commands', [])
    reward_cfg = task_config.get('reward_cfg', [])
    
    # 3. Create output directory
    os.makedirs(rollout_path, exist_ok=True)
    
    # 4. Build callbacks list
    callbacks = []
    
    # 4.1 Commands Callback (execute first for initialization)
    if commands:
        callbacks.append(CommandsCallback(commands))
    
    # 4.2 Judge Reset Callback (timeout check)
    if use_judge_reset:
        callbacks.append(JudgeResetCallback(max_steps))
    
    # 4.3 Rewards Callback (track rewards)
    if reward_cfg:
        callbacks.append(RewardsCallback(reward_cfg))
    
    # 4.4 Record Callback (video recording)
    if record_video:
        callbacks.append(
            RecordCallback(
                record_path=rollout_path,
                fps=fps,
                frame_type="pov"
            )
        )
    
    # 5. Create MinecraftSim environment
    env = MinecraftSim(
        action_type="env", 
        obs_size=obs_size, 
        render_size=render_size,
        preferred_spawn_biome=None,
        camera_config=camera_cfg,
        callbacks=callbacks,
        **kwargs
    )
    
    # 6. Create raw_action.jsonl file if requested
    if record_raw_action:
        raw_action_file_path = os.path.join(rollout_path, "raw_action.jsonl")
        with open(raw_action_file_path, 'w', encoding='utf-8') as f:
            # Initialize with empty entry for reset
            f.write(json.dumps({"raw_action": "", "step": 0}, ensure_ascii=False) + '\n')
    
    # 7. Save task configuration with metadata
    config_to_save = {
        'task_name': task_name,
        'yaml_path': yaml_path,
        **task_config
    }
    config_path = os.path.join(rollout_path, "task_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)
    
    return env, task_config


def save_episode_results(
    rollout_path: str,
    task_name: str,
    total_reward: float,
    steps_taken: int,
    success: bool,
    episode_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save episode results to JSON file.
    
    Args:
        rollout_path: Directory path where results are saved
        task_name: Name of the task
        total_reward: Total accumulated reward
        steps_taken: Number of steps taken
        success: Whether the task was successful
        episode_data: Optional detailed episode data
        
    Returns:
        Path to saved results file
    """
    results = {
        'task_name': task_name,
        'total_reward': float(total_reward),
        'steps_taken': int(steps_taken),
        'success': bool(success),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    if episode_data:
        results.update(episode_data)
    
    result_path = os.path.join(rollout_path, "episode_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return result_path


def run_mcu_task(
    yaml_path: str,
    agent,  # OpenHA agent
    rollout_path: str,
    obs_size: Tuple[int, int] = (224, 224),
    max_steps: int = 600,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single MCU task with the given agent.
    
    Args:
        yaml_path: Path to MCU task YAML file
        agent: OpenHA agent instance
        rollout_path: Directory path to save results
        obs_size: Observation image size
        max_steps: Maximum steps before timeout
        verbose: Whether to print verbose output
        
    Returns:
        Episode results dictionary
        
    Example:
        >>> from openagents.agents.openha import OpenHA
        >>> agent = OpenHA(model_path="...", output_mode="eager")
        >>> results = run_mcu_task(
        ...     yaml_path="tasks/combat/combat_zombies.yaml",
        ...     agent=agent,
        ...     rollout_path="./output/combat_zombies"
        ... )
        >>> print(f"Success: {results['success']}, Reward: {results['total_reward']}")
    """
    # 1. Initialize environment
    env, task_config = env_init_mcu(
        yaml_path=yaml_path,
        rollout_path=rollout_path,
        obs_size=obs_size,
        max_steps=max_steps,
        record_video=True,
        record_raw_action=True,
    )
    
    # 2. Reset agent
    task_name = Path(yaml_path).stem
    task_text = task_config['text']
    
    agent.reset(
        instruction=task_text,
        task_name=task_name,
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Task: {task_name}")
        print(f"Instruction: {task_text}")
        print(f"{'='*70}\n")
    
    # 3. Run episode
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    episode_steps = []
    raw_action_file_path = os.path.join(rollout_path, "raw_action.jsonl")
    
    while not done and step < max_steps:
        # Get action from agent
        action = agent.get_action(
            obs={'image': obs['image']},
            info={'pov': obs['image']},
            instruction=task_text,
            verbose=verbose,
        )
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        # Record step data
        step_data = {
            'step': step,
            'reward': float(reward),
            'response': agent.response,
            'policy_type': agent.policy_type,
        }
        episode_steps.append(step_data)
        
        # Save raw action
        with open(raw_action_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "raw_action": agent.response,
                "step": step,
                "reward": float(reward),
            }, ensure_ascii=False) + '\n')
        
        if verbose:
            print(f"Step {step}: Reward={reward:.2f}, Total={total_reward:.2f}, Policy={agent.policy_type}")
    
    # 4. Close environment
    env.close()
    
    # 5. Prepare results
    results = {
        'task_name': task_name,
        'task_text': task_text,
        'total_reward': float(total_reward),
        'steps_taken': step,
        'success': total_reward > 0,  # Simple success criterion
        'steps': episode_steps,
    }
    
    # 6. Save results
    save_episode_results(
        rollout_path=rollout_path,
        task_name=task_name,
        total_reward=total_reward,
        steps_taken=step,
        success=results['success'],
        episode_data={'steps': episode_steps},
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Episode finished!")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps Taken: {step}")
        print(f"Success: {results['success']}")
        print(f"{'='*70}\n")
    
    return results


def run_mcu_benchmark(
    agent,  # OpenHA agent
    tasks_dir: str,
    output_dir: str,
    task_categories: Optional[list] = None,
    obs_size: Tuple[int, int] = (224, 224),
    max_steps: int = 600,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run MCU benchmark on multiple tasks.
    
    Args:
        agent: OpenHA agent instance
        tasks_dir: Directory containing MCU task YAML files
        output_dir: Directory to save all results
        task_categories: List of categories to run (None = all categories)
        obs_size: Observation image size
        max_steps: Maximum steps per task
        verbose: Whether to print verbose output
        
    Returns:
        Benchmark results dictionary with statistics
        
    Example:
        >>> from openagents.agents.openha import OpenHA
        >>> agent = OpenHA(model_path="...", output_mode="eager")
        >>> results = run_mcu_benchmark(
        ...     agent=agent,
        ...     tasks_dir="openagents/assets/mcu_tasks",
        ...     output_dir="./output/mcu_benchmark",
        ...     task_categories=["combat", "mining_and_collecting"]
        ... )
        >>> print(f"Overall Success Rate: {results['overall_success_rate']:.2%}")
    """
    tasks_dir = Path(tasks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect tasks
    if task_categories is None:
        task_categories = [d.name for d in tasks_dir.iterdir() if d.is_dir()]
    
    benchmark_results = {
        'categories': {},
        'total_tasks': 0,
        'successful_tasks': 0,
        'total_reward': 0,
    }
    
    # Create date-based folder
    date_folder = time.strftime('%Y-%m-%d_%H-%M-%S')
    output_dir_with_date = output_dir / date_folder
    output_dir_with_date.mkdir(parents=True, exist_ok=True)
    
    for category in task_categories:
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# Category: {category}")
            print(f"{'#'*70}\n")
        
        # Get tasks in category
        yaml_files = list_mcu_tasks(str(tasks_dir), category)
        
        if not yaml_files:
            if verbose:
                print(f"No tasks found in category: {category}")
            continue
        
        category_results = []
        
        for yaml_path in yaml_files:
            task_name = Path(yaml_path).stem
            task_output_dir = output_dir_with_date / category / task_name
            
            try:
                result = run_mcu_task(
                    yaml_path=yaml_path,
                    agent=agent,
                    rollout_path=str(task_output_dir),
                    obs_size=obs_size,
                    max_steps=max_steps,
                    verbose=verbose,
                )
                category_results.append(result)
                
                if result['success']:
                    benchmark_results['successful_tasks'] += 1
                benchmark_results['total_reward'] += result['total_reward']
                benchmark_results['total_tasks'] += 1
                
            except Exception as e:
                print(f"Error running task {task_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate category statistics
        if category_results:
            avg_reward = sum(r['total_reward'] for r in category_results) / len(category_results)
            success_rate = sum(r['success'] for r in category_results) / len(category_results)
            
            benchmark_results['categories'][category] = {
                'num_tasks': len(category_results),
                'successful_tasks': sum(r['success'] for r in category_results),
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'tasks': category_results,
            }
            
            if verbose:
                print(f"\nCategory {category} Summary:")
                print(f"  Tasks: {len(category_results)}")
                print(f"  Success Rate: {success_rate:.2%}")
                print(f"  Average Reward: {avg_reward:.2f}")
    
    # Calculate overall statistics
    if benchmark_results['total_tasks'] > 0:
        benchmark_results['overall_success_rate'] = (
            benchmark_results['successful_tasks'] / benchmark_results['total_tasks']
        )
        benchmark_results['avg_reward'] = (
            benchmark_results['total_reward'] / benchmark_results['total_tasks']
        )
    else:
        benchmark_results['overall_success_rate'] = 0.0
        benchmark_results['avg_reward'] = 0.0
    
    # Save benchmark results
    result_path = output_dir_with_date / 'benchmark_results.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n{'#'*70}")
        print(f"# BENCHMARK SUMMARY")
        print(f"{'#'*70}")
        print(f"Total Tasks: {benchmark_results['total_tasks']}")
        print(f"Successful Tasks: {benchmark_results['successful_tasks']}")
        print(f"Overall Success Rate: {benchmark_results['overall_success_rate']:.2%}")
        print(f"Average Reward: {benchmark_results['avg_reward']:.2f}")
        print(f"Results saved to: {result_path}")
    
    return benchmark_results
