from lm_polygraph.utils.manager import UEManager

man = UEManager.load("/workdir/output/qa/{'path': 'mistralai/Mistral-7B-Instruct-v0.2', 'ensemble': False, 'mc': False, 'mc_seeds': None, 'dropout_rate': None, 'type': 'CausalLM', 'path_to_load_script': 'model/default_causal.py', 'load_model_args': {'device_map': 'auto'}, 'load_tokenizer_args': {}}/['trivia_qa', 'rc.nocontext']/2024-09-13/14-31-27/ue_manager_seed1")
print(man.stats['greedy_texts'])
print(man.stats['greedy_tokens'])


