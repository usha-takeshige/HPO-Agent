"""LLM provider abstractions and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class LLMProviderBase(ABC):
    """LLM インスタンスを提供するインターフェースを定義する。

    具象クラスは特定の LLM プロバイダー（Google, OpenAI, Anthropic 等）に対して
    このインターフェースを実装することで、Supervisor からの差し替えを可能にする。
    """

    @abstractmethod
    def get_llm(self, temperature: float = 0) -> BaseChatModel:
        """LLM インスタンスを返す。

        Args:
            temperature: サンプリング温度。0 は確定的、高いほど多様な出力。
        """
        ...


class GoogleLLMProvider(LLMProviderBase):
    """Google Gemini の LLM インスタンスを提供するプロバイダー。

    Args:
        api_key: Google API キー。
        model_name: 使用するモデル名（例: "gemini-1.5-pro"）。
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        """GoogleLLMProvider を初期化する。"""
        self._api_key = api_key
        self._model_name = model_name

    def get_llm(self, temperature: float = 0) -> BaseChatModel:
        """Google Gemini の LLM インスタンスを返す。

        Args:
            temperature: サンプリング温度。0 は確定的、高いほど多様な出力。
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=self._model_name,
            google_api_key=self._api_key,
            temperature=temperature,
        )


class OpenAILLMProvider(LLMProviderBase):
    """OpenAI GPT の LLM インスタンスを提供するプロバイダー。

    Args:
        api_key: OpenAI API キー。
        model_name: 使用するモデル名（例: "gpt-4o"）。
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        """OpenAILLMProvider を初期化する。"""
        self._api_key = api_key
        self._model_name = model_name

    def get_llm(self, temperature: float = 0) -> BaseChatModel:
        """OpenAI GPT の LLM インスタンスを返す。

        Args:
            temperature: サンプリング温度。0 は確定的、高いほど多様な出力。
        """
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self._model_name,
            api_key=self._api_key,  # type: ignore[arg-type]
            temperature=temperature,
        )


class AnthropicLLMProvider(LLMProviderBase):
    """Anthropic Claude の LLM インスタンスを提供するプロバイダー。

    Args:
        api_key: Anthropic API キー。
        model_name: 使用するモデル名（例: "claude-opus-4-6"）。
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        """AnthropicLLMProvider を初期化する。"""
        self._api_key = api_key
        self._model_name = model_name

    def get_llm(self, temperature: float = 0) -> BaseChatModel:
        """Anthropic Claude の LLM インスタンスを返す。

        Args:
            temperature: サンプリング温度。0 は確定的、高いほど多様な出力。
        """
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self._model_name,
            api_key=self._api_key,  # type: ignore[arg-type]
            temperature=temperature,
        )
