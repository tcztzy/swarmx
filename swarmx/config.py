from openai import AsyncOpenAI, OpenAI
from openai.types.chat_model import ChatModel
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings, env_prefix="SWARMX_"):  # type: ignore[call-arg]
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    default_model: ChatModel | str = "gpt-4o"

    @property
    def openai(self) -> OpenAI:
        return OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)

    @property
    def async_openai(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)


settings = Settings()  # type: ignore[call-arg]
