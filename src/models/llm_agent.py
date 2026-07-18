from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class LLMTradingAgent:
    """
    Local LLM integration for analyzing backtest results and market data.
    Based on the Qwen2.5-Instruct local loading pattern.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device_map: str = "auto"):
        self.model_id = model_id
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Lazily load the LLM weights."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading LLM {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype="auto", device_map=self.device_map
            )
            self._is_loaded = True
        except ImportError:
            logger.error("transformers or torch not installed. Cannot load LLM.")
            raise

    def generate_analysis(self, prompt: str, system_prompt: str = "", max_new_tokens: int = 512) -> str:
        self.load_model()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Strip the prompt tokens to get only the generated response
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def analyze_backtest(self, metrics: dict[str, Any], strategy_name: str) -> str:
        """Provide a natural language analysis of backtest metrics."""
        sys_prompt = "You are an expert quantitative trading analyst."
        
        prompt = (
            f"Please analyze the following backtest results for the '{strategy_name}' strategy.\n\n"
            f"Metrics:\n"
        )
        for k, v in metrics.items():
            prompt += f"- {k}: {v}\n"
            
        prompt += "\nDiscuss the strengths, weaknesses (e.g., drawdown, risk-adjusted returns), and potential overfitting risks."
        
        return self.generate_analysis(prompt, system_prompt=sys_prompt)

    def analyze_sentiment(self, news_df: pd.DataFrame, ticker: str) -> str:
        """Provide a summary of recent news sentiment for a ticker."""
        if news_df.empty:
            return f"No recent news for {ticker}."
            
        sys_prompt = "You are a financial news summarization AI."
        
        prompt = f"Analyze the recent headlines for {ticker} and provide a trading thesis (Bullish, Bearish, or Neutral):\n\n"
        for _, row in news_df.head(15).iterrows():
            prompt += f"- [{row['date']}] {row['title']}\n"
            
        return self.generate_analysis(prompt, system_prompt=sys_prompt)
