import asyncio
from typing import Any, Dict, List

from datetime import datetime as dt
import datetime

from pydantic import Field

from agentverse.logging import logger
from agentverse.environments import env_registry as EnvironmentRegistry
from agentverse.agents.simulation_agent.conversation import BaseAgent

from agentverse.environments.simulation_env.rules.twitter import TwitterRule as Rule
from agentverse.message import Message

from ..base import BaseEnvironment

from pydantic import validator
import mesa

import pickle
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.multiprocessing import Process, Queue, set_start_method, Manager
import torch
from tqdm import tqdm
import re
import yaml
from dateutil.relativedelta import relativedelta
# import mesa

MODEL = "/home/fanqi/llm_simulation/models/DPO/community_7/merged_ckp_4"
# MODEL = "NousResearch/Llama-2-7b-chat-hf"

def get_free_gpus():
    """
    获取空闲的 GPU 列表（显存占用率为 0 的 GPU）。
    """
    try:
        # 运行 nvidia-smi 并解析输出
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        memory_used = [int(x) for x in result.strip().split("\n")]
        free_gpus = [idx for idx, mem in enumerate(memory_used) if mem < 100]
        return free_gpus
    except Exception as e:
        print(f"Error while checking GPUs: {e}")
        return []
    
def llama_inference(prompts, batch_size, model_path, gpu_id, queue):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=2048)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    responses = []
    for i in tqdm(range(int(len(prompts)//batch_size))):
        if i != int(len(prompts)//batch_size)-1:
            batch_prompt = prompts[i*batch_size:(i+1)*batch_size]
        else:
            batch_prompt = prompts[i*batch_size:]
        with torch.no_grad():
            inputs = tokenizer(batch_prompt, padding=True, truncation=True, return_tensors="pt", max_length=2048).to(device)
            output = model.generate(**inputs, max_new_tokens=256, temperature=1.0, do_sample=True, repetition_penalty=1.3, pad_token_id = tokenizer.pad_token_id)
            output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
            processed_outputs = []
            for output in output_text:
                response_output = re.sub(r'\[INST\].*?\[/INST\]', '', output, flags=re.DOTALL).strip()
                processed_outputs.append(response_output)
        responses.extend(processed_outputs)
    queue.put((gpu_id, responses))

def parallel_llama_inference(prompts, batch_size, model_path):
    set_start_method('spawn', force=True) 
    free_gpus = get_free_gpus()
    print(f"Free GPUs found: {free_gpus}")
    num_gpus = len(free_gpus)
    prompts_per_gpu = len(prompts) // num_gpus
    manager = Manager()
    queue = manager.Queue()
    processes = []
    for i, gpu_id in enumerate(free_gpus):
        start_idx = i * prompts_per_gpu
        end_idx = start_idx + prompts_per_gpu
        if i != num_gpus - 1:
            prompts_gpu = prompts[start_idx:end_idx]
        else:
            prompts_gpu = prompts[start_idx:]
        p = Process(target=llama_inference, args=(prompts_gpu, batch_size, model_path, gpu_id, queue))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    all_responses = [None] * num_gpus
    while not queue.empty():
        gpu_id, responses = queue.get()
        all_responses[free_gpus.index(gpu_id)] = responses

    final_responses = [response for responses in all_responses for response in responses]
    return final_responses

@EnvironmentRegistry.register("twitter")
class TwitterEnvironment(BaseEnvironment):
    """
    Environment used in Observation-Planning-Reflection agent architecture.

    Args:
        agents: List of agents
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
        current_time
        time_delta: time difference between steps
        trigger_news: Dict, time(turn index) and desc of emergent events
    """

    agents: List[BaseAgent]
    rule: Rule
    max_turns: int = 10
    cnt_turn: int = 0
    last_messages: List[Message] = []
    rule_params: Dict = {}
    current_time: dt = dt.now()
    time_delta: int = 120
    trigger_news: Dict={}
    # tweet_db(firehose): store the tweets of all users; key: tweet_id, value: message
    tweet_db = {}
    output_path=""
    target="Metoo Movement"

    abm_model:mesa.Model = None
    # abm_model = None

    class Config:
        arbitrary_types_allowed = True
    # @validator("time_delta")
    # def convert_str_to_timedelta(cls, string):
    #
    #     return datetime.timedelta(seconds=int(string))

    def __init__(self, rule, **kwargs):
        rule_config = rule
        order_config = rule_config.get("order", {"type": "sequential"})
        visibility_config = rule_config.get("visibility", {"type": "all"})
        selector_config = rule_config.get("selector", {"type": "basic"})
        updater_config = rule_config.get("updater", {"type": "basic"})
        describer_config = rule_config.get("describer", {"type": "basic"})
        rule = Rule(
            order_config,
            visibility_config,
            selector_config,
            updater_config,
            describer_config,
        )

        super().__init__(rule=rule, **kwargs)
        self.rule.update_visible_agents(self)

        self.abm_model = kwargs['abm_model']

    async def step(self) -> List[Message]:
        """Run one step of the environment"""

        logger.info(f"Tick tock. Current time: {self.current_time}")

        # Get the next agent index
        agent_ids = self.rule.get_next_agent_idx(self)

        # Get the personal experience of each agent
        # await asyncio.gather(
        #             *[
        #                 self.agents[i].get_personal_experience()
        #                 for i in agent_ids
        #             ]
        # )   

        # Generate current environment description
        env_descriptions = self.rule.get_env_description(self)

        # check whether the news is a tweet; if so, add to the tweet_db
        self.check_tweet(env_descriptions)
        env_descriptions = self.rule.get_env_description(self)

        # Generate the next message
        # messages = await asyncio.gather(
        #     *[
        #         self.agents[i].astep(self.current_time, env_descriptions[i])
        #         for i in agent_ids
        #     ]
        # )
        selected_messages = None
        if len(agent_ids) > 0:
            prompts = [self.agents[i].generate_prompt(self.current_time, env_descriptions[i]) for i in agent_ids]
            responses = parallel_llama_inference(prompts, batch_size=4, model_path=MODEL)
            messages = [self.agents[i].step(self.current_time, env_descriptions[i], responses[index]) for index, i in enumerate(agent_ids)]

            # Some rules will select certain messages from all the messages
            selected_messages = self.rule.select_message(self, messages)
            self.last_messages = selected_messages
            self.print_messages(selected_messages)

        # Update opinion of mirror and other naive agents
        # update naive agents
        if self.abm_model is not None:
            self.abm_model.step()
            # then substitude the value of mirror using LLM results
            for i in agent_ids:
                self.abm_model.update_mirror(self.agents[i].name, self.agents[i].atts[-1])

        # Update the database of public tweets
        self.rule.update_tweet_db(self)
        print('Tweet Database Updated.')

        # Update the memory of the agents
        self.rule.update_memory(self)
        print('Agent Memory Updated.')

        # Update tweet page of agents
        self.rule.update_tweet_page(self)
        print('Tweet Pages Updated.')

        # TODO: Update the notifications(info box) of agents
        self.rule.update_info_box(self)
        print('Tweet Infobox Updated.')

        # Update the set of visible agents for each agent
        self.rule.update_visible_agents(self)
        print('Visible Agents Updated.')

        self.cnt_turn += 1

        # update current_time
        self.tick_tock()

        return selected_messages

    def print_messages(self, messages: List[Message]) -> None:
        for message in messages:
            if message is not None:
                logger.info(f"{message.sender}: {message.content}")

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        self.rule.reset()
        BaseAgent.update_forward_refs()
        for agent in self.agents:
            agent.reset(environment=self)

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns

    def tick_tock(self) -> None:
        """Increment the time"""
        # self.current_time = self.current_time + datetime.timedelta(
        #     seconds=self.time_delta
        # )
        self.current_time = self.current_time + relativedelta(days=1)

    def save_data_collector(self) -> None:
        """Output the data collector to the target file"""
        data = {}
        for agent in self.agents:
            data[agent.name] = agent.data_collector
        # naive agents in ABM model
        if self.abm_model is not None:
            opinion = {}
            for agent in self.abm_model.schedule.agents:
                opinion[agent.name] = agent.att[-1]
            data[f'opinion_results_{self.cnt_turn}'] = opinion
        output_path = self.output_path
        print('Output to {}'.format(output_path))
        if 'pkl' in output_path:
            with open(output_path,'wb') as f:
                pickle.dump(data, f)
        elif 'yaml' in output_path:
            with open(output_path, 'a') as f:
                yaml.dump(data, f)

    def check_tweet(self, env_descptions):
        cnt_turn = self.cnt_turn
        if len(env_descptions) > 0: 
            if 'posts a tweet' in env_descptions[0]:
                author = env_descptions[0][:env_descptions[0].index('posts a tweet')].strip()
                content = env_descptions[0]
                msg_lst = self.rule.update_tweet_db_for_news(self, author, content)
                self.rule.update_tweet_page_for_news(self, msg_lst)
                # del the trigger news
                self.trigger_news[cnt_turn]=""

    async def test(self, agent, context) -> List[Message]:
        """Run one step of the environment"""
        """Test the system from micro-level"""

        # Generate the next message
        prompt, message, parsed_response = await agent.acontext_test(context)

        return prompt, message, parsed_response