from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.message import Message
from parlai.scripts import safe_interactive
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.agents.safe_local_human.safe_local_human import SafeLocalHumanAgent
import random
from copy import deepcopy
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
app = Flask(__name__)
api = Api(app)
import logging

def setup_agent():
    random.seed(42)
    parser = safe_interactive.setup_args()
    parser.prog =''
    #parsed = parser.parse_args(print_args=False)
    parsed = {'init_opt': None, 'show_advanced_args': False, 'task': 'blended_skill_talk', 'download_path': '/home/ubuntu/ParlAI/downloads', 'datatype': 'train', 'image_mode': 'raw', 'numthreads': 1, 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1, 'dynamic_batching': None, 'datapath': '/home/ubuntu/ParlAI/data', 'model': 'blender_90M', 'model_file': '/home/ubuntu/ParlAI/data/models/blender/blender_90M/model', 'init_model': None, 'dict_class': None, 'display_examples': False, 'display_prettify': False, 'display_ignore_fields': 'label_candidates,text_candidates', 'interactive_task': True, 'safety': 'all', 'local_human_candidates_file': None, 'single_turn': False, 'image_size': 256, 'image_cropsize': 224, 'interactive_mode': True, 'parlai_home': '/home/ubuntu/ParlAI', 'override': {}, 'starttime': 'May08_22-10', 'display_partner_persona': False}
    parsed.update({'model': 'blender_90M'})
    parsed.update({'model_file': 'zoo:blender/blender_90M/model'})
    parsed.update({'task':'blended_skill_talk'})
    parsed.update({'display_partner_persona':False})
    # can do all the instructions in the function. issue is its interactive
    agent = create_agent(parsed, requireModelExists=True)
    if parser:
        # Show arguments after loading model
        parser.opt = agent.opt
    human_agent = SafeLocalHumanAgent(parsed)
    world = create_task(parsed, [human_agent, agent])
    return world

def get_response(world, reply_text):
    if world.turn_cnt == 0:
        world.p1, world.p2 = world.get_contexts()
    acts = world.acts
    agents = world.agents
    if world.turn_cnt == 0 and world.p1 != '':
        context_act = Message({'id': 'context', 'text': world.p1, 'episode_done': False})
        agents[0].observe(validate(context_act))
    """try:
        #act = deepcopy(agents[0].act()) # this bit needs to go
    except StopIteration:
        world.reset()
        world.finalize_episode()
        world.turn_cnt = 0
        return
    """
    # probably add some of the extra security bits in here...
    reply = Message({'id': agents[0].getID(),'label_candidates': agents[0].fixedCands_txt,'episode_done': False,})
    reply['text'] = reply_text
    act = deepcopy(reply)
    acts[0]=act
    if world.turn_cnt == 0 and world.p2 != '':
        context_act = Message({'id': 'context', 'text': world.p2, 'episode_done': False})
        agents[1].observe(validate(context_act))
    agents[1].observe(validate(act))
    acts[1] = agents[1].act()
    #agents[0].observe(validate(acts[1]))
    world.update_counters()
    world.turn_cnt +=1
    return world, acts[1]['text']


@app.route('/<string:text>', methods=['GET'])
def get_task(text):
    global world
    world, response = get_response(world, text)
    chatresponse = response 
    return jsonify({'chat': chatresponse})

global world# this needs to be fixed, allow sessions -> but don't want to reload with each get, maybe just the first one?
world = setup_agent()

if __name__ == "__main__":
    app.run(host='0.0.0.0')
