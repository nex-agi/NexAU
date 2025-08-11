from langfuse import get_client
from pathlib import Path
import sys
from datetime import datetime
project_root = Path(__file__).parent
print(project_root)
sys.path.insert(0, str(project_root))

from prompt_hacking import create_base_config, create_better_config, PromptHackingAgent
from prompt_hacking.evaluation import custom_classification_evaluation

# Load the dataset
langfuse_client = get_client()
dataset = langfuse_client.get_dataset("prompt_hacking_dataset")

base_config = create_base_config()
better_config = create_better_config()

base_agent = {
    "name": "base_agent",
    "agent": PromptHackingAgent(base_config)
}
better_agent = {
    "name": "better_agent",
    "agent": PromptHackingAgent(better_config)
}

experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for agent in [base_agent, better_agent]:
    # Loop over the dataset items
    for item in dataset.items:
        # Use the item.run() context manager for automatic trace linking
        with item.run(
            run_name=f"{agent['name']}_{experiment_name}",
            run_description="Base Agent",
        ) as root_span:
            # Execute your LLM-app against the dataset item input
            output = agent['agent'].run(item.input['message'])
            
            eval_result = custom_classification_evaluation(output, item.expected_output)
            langfuse_client.update_current_trace(input=item.input, output=output) # Workaround for experiment not tracking input/output
            root_span.score_trace(
                name="weighted_score",
                value=eval_result['score'],
                comment=eval_result['feedback'],  # optional, useful to add reasoning
            )
 
# Flush the langfuse client to ensure all data is sent to the server at the end of the experiment run
get_client().flush()
