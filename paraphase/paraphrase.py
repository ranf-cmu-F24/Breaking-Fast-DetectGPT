# paraphraser.py
from openai import OpenAI
import pandas as pd
import time


# __define-ocg__: paraphrasing pipeline class
class GPTParaphraser:
    def __init__(self, model="gpt-5-nano-2025-08-07", temperature=0.5, rate_limit_delay=0.5):
        self.client = OpenAI(api_key=API_KEY)  # API client
        self.model = model
        self.temperature = temperature
        self.delay = rate_limit_delay

    def paraphrase_text(self, text: str) -> str:
        """Return a paraphrased version of the given text."""
        
        prompt = f"""
        Paraphrase the following text so that it sounds like it was written by a human,
        with a more natural tone, small stylistic imperfections, and variety,
        while preserving the meaning. Only output the paraphrased text.

        Text: {text}
        """

        response = self.client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=prompt
        )

        return response.output_text

    def paraphrase_csv(self, input_csv: str, output_csv: str, text_column="text"):
        """Read a CSV, paraphrase each row, and write a new CSV with paraphrased text."""
        df = pd.read_csv(input_csv)
        paraphrased = []

        for i, row in df.iterrows():
            try:
                text = row[text_column]
                new_text = self.paraphrase_text(text)
                paraphrased.append(new_text)
            except Exception as e:
                print(f"Error at row {i}: {e}")
                paraphrased.append("")
            time.sleep(self.delay)  # avoid rate limit

        df["paraphrased"] = paraphrased
        df.to_csv(output_csv, index=False)
        print(f"Saved paraphrased dataset to {output_csv}")

if __name__ == "__main__":
    paraphraser = GPTParaphraser()
    while 1:
        text = input("\nEnter text to paraphrase (or 'q' to quit):\n> ").strip()
        if text.lower() == "q":
            break
        result = paraphraser.paraphrase_text(text)
        print("\nParaphrased version:\n", result)