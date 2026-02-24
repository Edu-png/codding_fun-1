"""
O matcher.py deve:

    receber dois textos

    transformar em vetores

    calcular similaridade

    retornar um score
"""
    
# Importando bibliotecas necessárias:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Simulando curriculo e vaga aleatórios:

cv_text = """
Eduardo Silva Coqueiro
Data Analyst / Junior Data Scientist

Summary
Data professional with hands-on experience in Python analytics and building data products. Comfortable with data cleaning, exploratory analysis, dashboards, and basic machine learning. Worked with stakeholders to translate business needs into metrics and reporting.

Core Skills
- Python (pandas, numpy), SQL (joins, CTEs, window functions)
- Data Cleaning, EDA, Feature Engineering
- Machine Learning (scikit-learn: classification, regression)
- Visualization (Power BI, Plotly, Matplotlib)
- Git, REST APIs

Experience
Data Analyst — Fintech SaaS (Remote) | 2024-2025
- Built automated reporting pipelines using Python + SQL, reducing manual reporting time by 60%
- Created KPI dashboards in Power BI for customer operations and finance teams
- Implemented a churn risk baseline model (logistic regression) and delivered monthly monitoring report

R&D Analyst — Forestry / Biotech | 2023-2024
- Consolidated field data from multiple regions and standardized templates for analysis
- Performed statistical analysis and created dashboards to support operational decisions

Education
BSc Biotechnology
Postgraduate studies in Data Science & Machine Learning (ongoing)
"""

job_text = """
Job Title: Data Analyst (Mid-level) — Remote (LATAM)

Responsibilities
- Analyze product and business data to produce insights and recommendations
- Build and maintain dashboards and automated reporting
- Write efficient SQL queries to extract and transform data
- Partner with product, operations, and finance to define KPIs
- Support A/B test analysis and metric design

Requirements
- Strong SQL (joins, CTEs, window functions)
- Python for analysis (pandas, numpy)
- Experience with BI tools (Power BI, Looker, Tableau)
- Understanding of statistics and experimentation
- Clear communication with non-technical stakeholders

Nice to have
- Basic machine learning knowledge (scikit-learn)
- Familiarity with cloud data warehouses (BigQuery/Snowflake)
"""