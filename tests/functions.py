from swarmx import Agent


def transfer_to_spanish_agent() -> Agent:
    return Agent(
        name="Spanish Agent",
        instructions="You only speak Spanish.",
    )


def print_account_details(context_variables: dict):
    """Simple function to print account details."""
    user_id = context_variables.get("user_id", None)
    name = context_variables.get("name", None)
    return f"Account Details: {name} {user_id}"
