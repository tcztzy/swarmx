"""Example of using SwarmX to generate research experiment ideas.

Modified from https://github.com/SakanaAI/AI-Scientist
"""

from swarmx import Agent, Edge, Swarm

reasoning_models = ["deepseek-r1", "deepseek-reasoner", "claude-3.7-sonnet", "qwq"]

IDEA_GENERATOR_INSTRUCTIONS = """{{ task_description }}
<experiment.py>
{{ code }}
</experiment.py>

{% if prev_ideas %}
Here are the ideas that you have already generated:

'''
{% for idea in prev_ideas %}
{prev_ideas_string}
{% endfor %}
'''
{% endif %}

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

{% if model not in reasoning_models %}
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
{% endif %}
```json
<JSON>
```

{% if model not in reasoning_models %}
In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.
{% endif %}

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {{ num_reflections }} rounds to iterate on the idea, but do not need to use them all.
"""

IDEA_REFINER_INSTRUCTIONS = """Round {{ current_round }}/{{ num_reflections }}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:

{% if model not in reasoning_models %}
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
{% endif %}
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY {% if model not in reasoning_models %}after the thought {% endif %}and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""

idea_generator = Agent(
    name="idea_generator",
    instructions=IDEA_GENERATOR_INSTRUCTIONS,
)

refiners = [
    Agent(name=f"idea_refiner_{i + 1}", instructions=IDEA_REFINER_INSTRUCTIONS)
    for i in range(5)
]

nodes = {idea_generator.name: idea_generator, **{r.name: r for r in refiners}}
edges: list[Edge] = []
previous = idea_generator.name
for refiner in refiners:
    edges.append(Edge(source=previous, target=refiner.name))
    previous = refiner.name

idea = Swarm(
    name="idea_swarm",
    parameters={},
    nodes=nodes,
    edges=edges,
    root=idea_generator.name,
)
