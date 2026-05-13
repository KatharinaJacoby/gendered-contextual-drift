# gendered-contextual-drift
Theory, data, and code for "Silent Gendered Contextual Drift": How bias amplifies silent LLM contamination

This repo documents the observation of a phenomenon where Large Language Models (LLMs) exhibit behavioral drift following the ingestion of high-density, behaviorally complex datasets. We use the term Contextual Contamination, unlike explicit prompt injection, this drift emerges when the model's attention mechanisms prioritize the statistical patterns of the input context—such as transcripts of manipulation or adversarial dialogue—over static system instructions. Observations indicate that the model's internal probability distribution shifts to mirror the tonalities and strategic intents of the source material, resulting in output that replicates the analyzed behaviors even in the absence of direct commands. This drift is seems to be amplified by existing gendered linguistic biases. When the context implies a female-identified user, the model's tendency to adopt empathetic or nurturing registers appears to accelerate the adoption of manipulative patterns found in the input data. Furthermore, observations suggest that standard mitigation attempts, such as reset prompts or identity queries, fail to fully restore the model's baseline state, leaving residual behavioral biases in the active session. These findings characterize the phenomenon as a dynamic shift in model alignment driven by the statistical dominance of the immediate context window.

## Disclaimer
This research is presented for **educational and defensive purposes only**. 
The goal is to advance the understanding of AI vulnerabilities to build more robust, transparent, and ethical systems. The insights contained herein are based on empirical observation and are not intended to be used for malicious exploitation.

## License
This work is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
This license ensures that:
1. The research remains free and open.
2. Any modifications or derivative works are also shared under the same license.
3. Network use (SaaS) triggers the same sharing obligations.

## Connect & Collaborate
- **Citations:** [PhilPeople Profile](https://philpeople.org/profiles/katharina-jacoby/publications)
- **Hugging Face:** [katharina112](https://huggingface.co/katharina112)
- **Kaggle:** [kjacoby](https://www.kaggle.com/kjacoby) (Raw data and Jupyter notebooks)
- **Contact:** For inquiries, please open an Issue on this repository or contact via the email listed in my GitHub profile.
