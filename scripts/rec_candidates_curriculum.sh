# scripts/run_curriculum.sh
set -x

MODEL_PATH=/home/jiangjunguang.jjg/LLM_models/qwen7b-instruct-1m
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=123457

# 定义候选数据集
CANDIDATE_FILES=("3candidates.parquet" "5candidates.parquet" "7candidates.parquet" "10candidates.parquet")
EXPERIMENT_NAME="beauty-candidates-cot-new"

for candidate in "${CANDIDATE_FILES[@]}"; do
    echo "Starting training for ${candidate}"

    TRAIN_FILE="/home/jiangjunguang.jjg/code_linlin/Logic-RL-main/data/amazon_beauty_candidates/${candidate}"
    VAL_FILE="/home/jiangjunguang.jjg/code_linlin/Logic-RL-main/data/amazon_beauty_candidates/test.parquet" # 固定验证集

    # 定义当前阶段的实验名称
    CURRENT_EXPERIMENT_NAME="${EXPERIMENT_NAME}-${candidate}"

    # 构建并执行训练命令
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=reinforce_plus_plus \
        data.train_files=${TRAIN_FILE} \
        data.val_files=${VAL_FILE} \
        data.train_batch_size=8 \
        data.val_batch_size=8 \
        data.max_prompt_length=10240 \
        data.max_response_length=10240 \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_micro_batch_size=8 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.grad_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=40 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=40 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['wandb'] \
        trainer.project_name='0410-beauty-candidates-cot' \
        trainer.experiment_name=${CURRENT_EXPERIMENT_NAME} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.default_local_dir=/home/jiangjunguang.jjg/code_linlin/Logic-RL-main/local_dir/0410-beauty-cot/${CURRENT_EXPERIMENT_NAME} \
        trainer.default_hdfs_dir=null \
        trainer.save_freq=30 \
        trainer.test_freq=10 \
        trainer.total_epochs=5 $@ 2>&1 | tee reinforce.log

    # 更新模型路径为当前阶段的检查点路径 
    # TODO
    MODEL_PATH="/home/jiangjunguang.jjg/code_linlin/Logic-RL-main/local_dir/0410-beauty-cot/${CURRENT_EXPERIMENT_NAME}/actor"
done

echo "Curriculum learning finished!"