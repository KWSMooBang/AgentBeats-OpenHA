#! /bin/zsh

python examples/rollout_mcu_benchmark.py \
    --category "crafting" \
    --max_steps 600 \
    --output_mode grounding \
    --output_format grounding_coa \
    --model_id CrossAgent-qwen2vl-7b \
    --model_path /workspace/models/CrossAgent/ \
    --sam_path /workspace/woosung/AgentBeats-OpenHA/external/SAM2/checkpoints \
    --grounding_policy_path /workspace/models/ROCKET-1.12w_EMA \
    --motion_policy_path /workspace/models/Motion-Policy/model.safetensors \
    --vlm_client_mode "online" \
    --maximum_history_length 5 \