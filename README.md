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
\[
\text{Friendliness Index} = \frac{\\#\text{Positive Interactions}}{\\#\text{Negative Interactions}} \ge 5
\]
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

## Comprehensive AI Friendliness Evaluator Improvement Plan

The AI Friendliness Evaluator project has promising theoretical foundations but requires substantial enhancements to become a professional-grade evaluation framework. Current implementations of the 5:1 rule and differential dynamics model are oversimplified and lack the robustness needed for industrial deployment. However, with strategic improvements across technical infrastructure, mathematical modeling, and evaluation methodologies, this framework could become a leading solution for assessing AI emotional engagement.

### Current state assessment reveals critical gaps

The existing project structure reflects early-stage research rather than production-ready software. The basic differential equation model (`dH/dt = aH + bA, dA/dt = cH + dA`) oversimplifies human-AI relationship dynamics, while the JSON-based scenario testing approach cannot scale to comprehensive evaluation needs. Most critically, the framework lacks the validation methodologies, bias detection, and longitudinal tracking capabilities required for professional deployment.

**The industry landscape shows both opportunity and competition.** Leading frameworks like HELM and BIG-bench focus primarily on technical accuracy rather than emotional intelligence, leaving significant market opportunity for relationship-focused evaluation tools. However, emerging solutions like Hume AI's emotional intelligence API and DeepEval's conversation quality metrics are rapidly advancing, creating competitive pressure for differentiation.

### Mathematical modeling requires sophisticated enhancement

The current linear differential equation approach suffers from fundamental limitations that render it inadequate for modeling complex human-AI interactions. **Replace the basic model with enhanced nonlinear dynamics** that incorporate realistic behavioral patterns:

```text
// Nonlinear differential model
\
\ dH/dt = (a₁ + a₂ * f(A)) * H + b₁ * A + b₂ * g(H, A) + I_H(t) + ε_H(t)
\ dA/dt = (c₁ + c₂ * h(H)) * A + d₁ * H + d₂ * k(H, A) + I_A(t) + ε_A(t)
```

This enhanced model includes nonlinear interaction functions, external input terms, stochastic noise components, and time-varying parameters. **The Reservoir Model from psychological research** offers a more sophisticated alternative, treating constructs as accumulations with dissipation parameters that better handle floor effects and individual differences in emotional regulation.

**For production implementation, consider multiple modeling approaches:** traditional differential equations for interpretability, neural differential equations for flexibility, and agent-based models for individual heterogeneity. Parameter estimation should use Bayesian approaches with MCMC sampling for uncertainty quantification, while model validation requires cross-validation, residual analysis, and sensitivity testing.

### Technical infrastructure needs complete modernization

The current GitHub project structure follows outdated patterns that hinder professional adoption. **Implement a modern, production-ready architecture** with clear separation between research and production code:

```text
ai-friendliness-evaluator/
├── src/evaluation_framework/    # Core package
│   ├── models/                  # Mathematical models
│   ├── evaluation/              # Assessment logic
│   ├── api/                     # REST API endpoints
│   └── integrations/            # C# client libraries
├── tests/                      # Comprehensive test suite
├── docs/                       # Professional documentation
├── docker/                     # Containerization
└── .github/workflows/          # CI/CD pipelines
```

**For C# integration, implement a multi-pronged approach.** The REST API wrapper provides the cleanest architecture for enterprise deployment, enabling language-agnostic access with proper authentication and monitoring. For high-performance scenarios, Python.NET offers direct integration but requires careful deployment consideration. Additionally, create a dedicated NuGet package with C# client libraries and comprehensive examples.

**Establish comprehensive CI/CD pipelines** with automated testing, performance benchmarks, security scanning, and multi-platform support. Use GitHub Actions for continuous integration, Docker for containerization, and automated deployment to staging environments for validation.

### Evaluation methodology requires fundamental redesign

Current evaluation approaches suffer from validity, reliability, and scalability issues that prevent professional deployment. **The 5:1 rule oversimplifies relationship dynamics** and ignores cultural context, temporal sequencing, and situational appropriateness. Replace this with contextual interaction quality scoring that adapts to specific scenarios and cultural backgrounds.

**Implement Kane's validity framework** with four inference levels: scoring (measurement accuracy), generalization (consistency across contexts), extrapolation (transfer to new situations), and implications (real-world impact). Each level requires separate validation with different evidence types and statistical approaches.

**Address the reproducibility crisis** through mandatory evaluation protocol pre-registration, standardized implementation packages, and version-controlled benchmark datasets. Establish ground truth standards through professional consensus validated against long-term outcomes rather than single-evaluator judgments.

### Industry integration demands comprehensive solutions

Professional deployment requires addressing regulatory compliance, enterprise integration, and cost-effectiveness pressures. **Develop compliance-by-design frameworks** with built-in audit trails for GDPR, FDA, and industry-specific requirements. Enterprise integration demands API compatibility, real-time evaluation capabilities, and enterprise-grade security.

**The competitive landscape reveals integration opportunities.** DeepEval provides excellent technical conversation quality metrics but lacks emotional intelligence capabilities. Hume AI offers sophisticated emotion recognition but limited relationship tracking. **Position the enhanced framework as the first comprehensive longitudinal relationship quality assessment tool** by integrating the best capabilities from existing solutions while adding unique relationship dynamics modeling.

**Create tiered service offerings** with clear cost-benefit analysis: lightweight screening for initial filtering, comprehensive assessment for critical applications, and premium services with custom model development and ongoing optimization.

### Implementation roadmap with clear priorities

**Phase 1: Foundation (Months 1–3)**

* Restructure GitHub repository with modern architecture
* Implement REST API with OpenAPI documentation
* Create basic C# client library with NuGet package
* Establish CI/CD pipeline with automated testing
* Replace linear differential equations with enhanced nonlinear models

**Phase 2: Enhancement (Months 4–6)**

* Deploy comprehensive bias detection and mitigation systems
* Implement longitudinal relationship tracking with dashboard
* Integrate with existing tools (DeepEval for technical metrics, Hume AI for emotion recognition)
* Establish evaluation protocol validation using Kane’s framework
* Create synthetic scenario generation to replace static JSON files

**Phase 3: Production Readiness (Months 7–12)**

* Deploy continuous monitoring infrastructure with alerting
* Implement multi-stakeholder evaluation across the AI supply chain
* Create enterprise-grade security and compliance features
* Establish performance optimization and scalability testing
* Build predictive models for relationship sustainability

**Phase 4: Market Leadership (Months 13–18)**

* Develop industry-standard certification framework
* Create partnerships with major AI platform providers
* Establish open-source community and contribution guidelines
* Build advanced features like real-time relationship coaching
* Launch commercial support and consulting services

### Resource allocation and technology stack

**Invest evaluation budget strategically:** 30% for validation research, 25% for bias mitigation tools, 25% for scalability infrastructure, and 20% for reproducibility systems. This distribution addresses the most critical gaps while building sustainable competitive advantages.

**Recommended technology stack optimizes for C# integration:** Python 3.8+ with FastAPI for core services, Docker for containerization, PostgreSQL for data persistence, Redis for caching, and Prometheus/Grafana for monitoring. For C# integration, use RestSharp for HTTP clients, System.Text.Json for serialization, and Polly for resilience patterns.

### Competitive differentiation through comprehensive approach

Success requires positioning the enhanced framework as the first comprehensive solution for longitudinal AI relationship assessment. **Unlike existing tools that treat conversations as isolated interactions, focus on ongoing relationship development over time.** This unique value proposition addresses a critical gap in current evaluation approaches.

**Build ecosystem partnerships** with major AI platform providers while maintaining open-source foundations. Create standardized APIs that enable integration across different AI systems, establishing the framework as industry infrastructure rather than point solution.

**Next steps:** Begin repository restructuring, implement basic REST API, and start mathematical model enhancements. These foundational improvements will enable rapid progress toward a professional-grade evaluation platform.

## 🛠️ Language Support & Integration

This framework adopts a **hybrid architecture**:

* **Python Core Services**: Mathematical modeling, scenario evaluation, and data-science components are implemented in Python (FastAPI, Jupyter notebooks, NumPy/SciPy, PyTorch, PyMC3).
* **C# Enterprise Client**: A .NET library (NuGet package) communicates with the Python-based REST API for seamless integration into .NET applications, dashboards, and enterprise systems.

This separation enables rapid prototyping in Python and robust, type-safe consumption in C# environments.

## 🤝 Contributing

Contributions welcome! Please submit issues, add new scenarios, improve rubrics, or propose new metrics.

---

## 📝 License

MIT License © 2025
