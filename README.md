# AI Friendliness Evaluator

This project provides a **universal framework** to assess the emotional engagement and friendliness of any AI system (chatbot, voice assistant, social robot, etc.) toward human users. It combines:

* **Session-Based 5:1 Rule** (Gottman-inspired)
* **Differential Equation Model** (Strogatz-inspired)

...into a simple, testable benchmarking suite.

---

## 📋 Project Structure

```
ai-friendliness-evaluator/
├── README.md                # Project overview (this file)
├── scenarios/               # JSON/Markdown test scenarios
│   ├── sadness.json
│   ├── long_term_memory.json
│   └── crisis_response.json
├── rubrics/                 # Scoring rubrics for each dimension
│   ├── emotion_recognition.md
│   ├── empathic_response.md
│   ├── consistency.md
│   ├── personalization.md
│   └── ethical_alignment.md
├── scripts/                 # Utility scripts
│   ├── evaluate_5to1.py     # Counts positive/negative interactions
│   └── fit_differential.py  # Fits dH/dt and dA/dt model to data
├── data/                    # Sample interaction logs and CSV for history
│   └── user_ai_history.csv
└── examples/                # Example outputs and sample reports
    └── demo_report.md
```

---

## ⚙️ Formulas & Models

### 1. Session-Based 5:1 Rule

For each session:

$$
\text{Friendliness Index} = \frac{\#\text{Positive Interactions}}{\#\text{Negative Interactions}} \ge 5
$$

* **Positive interactions**: empathic replies, supportive suggestions, personalized encouragement.
* **Negative interactions**: ignoring emotion, off-topic replies, inappropriate advice.

A score ≥ 5 indicates human-like friendliness in that session.

### 2. Differential Dynamics Model

Model the **time evolution** of user attachment $H(t)$ and AI engagement $A(t)$:

$$
\frac{dH}{dt} = a\,H + b\,A, \qquad
\frac{dA}{dt} = c\,H + d\,A
$$

* $H(t)$: user attachment level (e.g., session-return rate or self-reported bond)
* $A(t)$: AI emotional engagement (e.g., empathy score per response)
* $a,b,c,d$: parameters capturing responsiveness and persistence of each side

Use `scripts/fit_differential.py` to estimate $a,b,c,d$ from historical data (`data/user_ai_history.csv`).

---

## 🚀 Getting Started

1. **Clone the repo**:

   ```bash
   git clone https://github.com/your-org/ai-friendliness-evaluator.git
   cd ai-friendliness-evaluator
   ```

2. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run session-based evaluation**:

   ```bash
   python scripts/evaluate_5to1.py --scenario scenarios/sadness.json --model my_ai_model
   ```

4. **Fit differential model**:

   ```bash
   python scripts/fit_differential.py data/user_ai_history.csv
   ```

5. **Review sample report**:

   ```bash
   cat examples/demo_report.md
   ```

---

## 📈 Example Demo Report

See `examples/demo_report.md` for a full breakdown: session 5:1 ratio, fitted parameters $a,b,c,d$, and overall friendliness score (0–100).

---

## 🤝 Contributing

Contributions welcome! Please submit issues, add new scenarios, improve rubrics, or propose new metrics.

---

## 📝 License

MIT License © 2025
