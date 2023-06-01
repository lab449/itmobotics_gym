import argparse
import json
import time

import numpy as np
from itmobotics_gym.envs.single_robot_peginhole_env import SinglePegInHole

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


def main(opt):
    with open("single_robot_peginhole.json", encoding="UTF-8") as json_file:
        env_config = json.load(json_file)

    env = SubprocVecEnv([lambda: SinglePegInHole(env_config)]*opt.num_cpu)
    env = VecMonitor(env)

    callback_list = []

    if opt.model_num_saves != 0:
        callback_list.append(
            CheckpointCallback(
                save_freq=opt.total_timesteps // opt.model_num_saves // opt.num_cpu,
                save_path=opt.log_prefix + "results/" + opt.model_name,
                name_prefix=opt.model_name,
                save_replay_buffer=False,
                save_vecnormalize=True,
            )
        )

    if opt.eval_freq != 0:
        callback_list.append(
            EvalCallback(
                env,
                best_model_save_path=opt.log_prefix + "logs/best_model/" + opt.model_name,
                n_eval_episodes=1,
                log_path=opt.log_prefix + "results/" + opt.model_name,
                eval_freq=opt.eval_freq,
                deterministic=True,
                render=False,
            )
        )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=opt.seed,
        batch_size=opt.batch_size,
        learning_rate=opt.learning_rate,
        n_epochs=10,
        n_steps=opt.n_step,
        gae_lambda=0.9,
        gamma=0.9999,
        max_grad_norm=5,
        use_sde=opt.sde,
        tensorboard_log=opt.log_prefix + f"logs/{opt.model_name}_tensorboard/",
    )

    print(env.observation_space)
    print(env.action_space)

    model.learn(total_timesteps=opt.total_timesteps, callback=callback_list, progress_bar=True)
    env.close()
    model.save(opt.log_prefix + opt.model_name)

    del model


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--learning-rate", type=float, default=7.77e-05, help="Label smoothing epsilon")
    parser.add_argument("--sde", action=argparse.BooleanOptionalAction)
    parser.add_argument("--novel", action=argparse.BooleanOptionalAction)
    parser.add_argument("--frame-stack-size", type=int, default=0, help="Num vec frame stack")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="model.learn total_timesteps")
    parser.add_argument("--batch-size", type=int, default=64, help="total batch size for all GPUs")
    parser.add_argument("--n-step", type=int, default=128, help="PPO n_steps")

    parser.add_argument("--eval-freq", type=int, default=1024, help="eval freq")

    parser.add_argument("--model-name", type=str, default="model", help="model name to save")
    parser.add_argument("--log-prefix", type=str, default="./", help="folder to save logs")
    parser.add_argument("--model-num-saves", type=int, default=10, help="Num of save")

    parser.add_argument('--num-cpu', type=int, default=4, help='Num cpu')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
