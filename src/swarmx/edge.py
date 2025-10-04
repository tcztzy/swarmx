"""Edge."""

from pydantic import BaseModel


class Edge(BaseModel, frozen=True, use_attribute_docstrings=True):
    """Edge in the agent graph.

    Using this when you need to create an edge for transferring conversation control from source agent to target agents. All existing agents are list in tools before this one.

    Examples:
    - Context: User needs to transfer from a general assistant to a specialized Python programming agent
        user: "I need help writing a complex Python script for data processing"
        assistant: "I'll create an edge from the current assistant to the PythonExpert agent, as this requires specialized programming expertise beyond my general capabilities."
    - Context: Data analysis task requires visualization expertise
        user: "I have analyzed the sales data, now I need to create interactive charts and dashboards"
        assistant: "I'll create an edge from DataAnalyst to VisualizationSpecialist agent to handle the chart creation and dashboard development."
    - Context: Content creation task requires multiple specialized agents
        user: "I need to create a technical blog post, then optimize it for SEO and social media sharing"
        assistant: "I'll create edges from ContentWriter to SEOSpecialist and SocialMediaManager agents to handle the optimization and distribution phases."
    - Context: API integration requires security review
        user: "I've built the API integration, but need security validation before deployment"
        assistant: "I'll create an edge from WebIntegration to SecurityReviewer agent to ensure the implementation follows security best practices."

    """

    source: str | tuple[str, ...]
    """Name of the source node, if is array of string, which means transferring to target need all sources done."""
    target: str
    """Name of the target node, could be agent's name or (tool/common expression language) which returns agent names."""
