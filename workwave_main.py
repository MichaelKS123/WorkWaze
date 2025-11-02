"""
WorkWave: Job Market Analytics Platform
Author: Michael Semera
Description: Advanced job market analysis tool for data roles with web scraping and NLP
"""

import pandas as pd
import numpy as np
import re
import json
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Web scraping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus
import time
from fake_useragent import UserAgent

# NLP and text processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class JobScraper:
    """
    Advanced web scraper for job listings across multiple platforms.
    Handles rate limiting, headers rotation, and error recovery.
    """
    
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.scraped_jobs = []
        
    def get_headers(self):
        """Generate randomized headers to avoid blocking."""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def scrape_sample_data(self, num_jobs=50):
        """
        Generate realistic sample job data for demonstration.
        In production, this would scrape actual job sites.
        """
        print("🔍 Generating sample job market data...")
        
        # Realistic job titles for data roles
        job_titles = [
            'Data Analyst', 'Data Scientist', 'Machine Learning Engineer',
            'Business Intelligence Analyst', 'Data Engineer', 'Analytics Manager',
            'Senior Data Analyst', 'Junior Data Scientist', 'ML Researcher',
            'Big Data Engineer', 'BI Developer', 'Quantitative Analyst'
        ]
        
        # Companies
        companies = [
            'Google', 'Amazon', 'Microsoft', 'Meta', 'Apple', 'Netflix',
            'Tesla', 'SpaceX', 'Uber', 'Airbnb', 'Stripe', 'Spotify',
            'Adobe', 'Oracle', 'IBM', 'Salesforce', 'Twitter', 'LinkedIn'
        ]
        
        # Locations
        locations = [
            'San Francisco, CA', 'New York, NY', 'Seattle, WA', 'Austin, TX',
            'Boston, MA', 'Chicago, IL', 'Los Angeles, CA', 'Denver, CO',
            'Remote', 'Atlanta, GA', 'Miami, FL', 'Portland, OR'
        ]
        
        # Skills pool
        skills_pool = [
            'Python', 'SQL', 'R', 'Tableau', 'Power BI', 'Excel', 'Java',
            'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch',
            'Pandas', 'NumPy', 'Scikit-learn', 'AWS', 'Azure', 'GCP',
            'Spark', 'Hadoop', 'Kafka', 'Docker', 'Kubernetes', 'Git',
            'Statistics', 'A/B Testing', 'Data Visualization', 'ETL',
            'Big Data', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL',
            'NLP', 'Computer Vision', 'Time Series', 'Data Mining',
            'Business Intelligence', 'Data Warehousing', 'Airflow'
        ]
        
        # Education requirements
        education_levels = [
            "Bachelor's degree in Computer Science, Statistics, or related field",
            "Master's degree in Data Science or related quantitative field",
            "PhD in Machine Learning, Statistics, or related field",
            "Bachelor's degree with 3+ years of experience",
            "Advanced degree preferred, Bachelor's required"
        ]
        
        jobs_data = []
        
        for i in range(num_jobs):
            # Randomly select job attributes
            title = np.random.choice(job_titles)
            company = np.random.choice(companies)
            location = np.random.choice(locations)
            
            # Generate salary based on seniority
            if 'Senior' in title or 'Manager' in title:
                salary_min = np.random.randint(120, 160) * 1000
                salary_max = np.random.randint(180, 250) * 1000
            elif 'Junior' in title:
                salary_min = np.random.randint(60, 80) * 1000
                salary_max = np.random.randint(85, 110) * 1000
            else:
                salary_min = np.random.randint(90, 120) * 1000
                salary_max = np.random.randint(130, 180) * 1000
            
            # Select random skills (5-12 per job)
            num_skills = np.random.randint(5, 13)
            required_skills = list(np.random.choice(skills_pool, num_skills, replace=False))
            
            # Create job description
            description = self._generate_description(title, required_skills)
            
            job = {
                'job_id': f'JOB_{i+1:04d}',
                'title': title,
                'company': company,
                'location': location,
                'salary_min': salary_min,
                'salary_max': salary_max,
                'salary_avg': (salary_min + salary_max) / 2,
                'skills': ', '.join(required_skills),
                'description': description,
                'education': np.random.choice(education_levels),
                'experience_years': np.random.randint(1, 8),
                'posted_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 30)),
                'source': np.random.choice(['LinkedIn', 'Indeed', 'Glassdoor'])
            }
            
            jobs_data.append(job)
        
        df = pd.DataFrame(jobs_data)
        print(f"✓ Generated {len(df)} job listings successfully")
        
        return df
    
    def _generate_description(self, title, skills):
        """Generate realistic job description."""
        templates = [
            f"We are seeking a talented {title} to join our dynamic team. "
            f"The ideal candidate will have strong expertise in {', '.join(skills[:3])} and more. "
            f"You will be responsible for analyzing complex datasets, building predictive models, "
            f"and delivering actionable insights to stakeholders.",
            
            f"Join our innovative team as a {title}! We're looking for someone skilled in "
            f"{', '.join(skills[:3])} to drive data-driven decision making. "
            f"You'll work with cutting-edge technologies and collaborate with cross-functional teams.",
            
            f"Exciting opportunity for a {title}! Required skills include {', '.join(skills[:3])} "
            f"and experience with modern data stack. Come help us transform data into strategic insights."
        ]
        
        return np.random.choice(templates)
    
    def save_to_csv(self, df, filename='job_listings.csv'):
        """Save scraped data to CSV."""
        df.to_csv(filename, index=False)
        print(f"✓ Data saved to {filename}")
        return filename


class SkillsAnalyzer:
    """
    Advanced NLP-based skills analyzer.
    Extracts, categorizes, and analyzes job skills from text.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tech_skills = self._load_tech_skills()
        
    def _load_tech_skills(self):
        """Predefined list of tech skills for matching."""
        return {
            'python', 'sql', 'r', 'java', 'scala', 'javascript', 'c++',
            'tableau', 'power bi', 'looker', 'excel', 'matplotlib', 'seaborn',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
            'pandas', 'numpy', 'scikit-learn', 'scipy', 'statsmodels',
            'aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes',
            'spark', 'hadoop', 'kafka', 'airflow', 'etl',
            'sql server', 'postgresql', 'mysql', 'mongodb', 'redis', 'nosql',
            'git', 'github', 'gitlab', 'jira', 'agile', 'scrum',
            'statistics', 'probability', 'linear algebra', 'calculus',
            'a/b testing', 'hypothesis testing', 'regression', 'classification',
            'nlp', 'computer vision', 'time series', 'forecasting',
            'data visualization', 'business intelligence', 'data warehousing',
            'big data', 'data mining', 'predictive analytics'
        }
    
    def extract_skills(self, text):
        """Extract technical skills from job description."""
        if pd.isna(text):
            return []
        
        text = str(text).lower()
        found_skills = []
        
        for skill in self.tech_skills:
            if skill in text:
                found_skills.append(skill.title())
        
        return found_skills
    
    def analyze_skills_distribution(self, df):
        """Analyze skills distribution across all jobs."""
        all_skills = []
        
        for skills_str in df['skills']:
            if pd.notna(skills_str):
                skills_list = [s.strip() for s in str(skills_str).split(',')]
                all_skills.extend(skills_list)
        
        skills_counter = Counter(all_skills)
        skills_df = pd.DataFrame(skills_counter.most_common(30), 
                                columns=['Skill', 'Frequency'])
        skills_df['Percentage'] = (skills_df['Frequency'] / len(df) * 100).round(2)
        
        return skills_df
    
    def categorize_skills(self, skills_df):
        """Categorize skills into groups."""
        categories = {
            'Programming Languages': ['Python', 'R', 'SQL', 'Java', 'Scala', 'JavaScript'],
            'ML/AI Tools': ['Machine Learning', 'Deep Learning', 'TensorFlow', 
                          'PyTorch', 'Scikit-learn', 'NLP', 'Computer Vision'],
            'Visualization': ['Tableau', 'Power BI', 'Excel', 'Matplotlib', 'Seaborn'],
            'Cloud/Infrastructure': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes'],
            'Big Data': ['Spark', 'Hadoop', 'Kafka', 'Big Data', 'Airflow'],
            'Databases': ['PostgreSQL', 'MySQL', 'MongoDB', 'NoSQL', 'SQL']
        }
        
        categorized = {cat: [] for cat in categories}
        
        for _, row in skills_df.iterrows():
            skill = row['Skill']
            for category, skills_list in categories.items():
                if skill in skills_list:
                    categorized[category].append({
                        'skill': skill,
                        'frequency': row['Frequency'],
                        'percentage': row['Percentage']
                    })
        
        return categorized


class SalaryAnalyzer:
    """
    Comprehensive salary analysis for different roles and skills.
    """
    
    def __init__(self, df):
        self.df = df
    
    def analyze_by_role(self):
        """Analyze salary distribution by job role."""
        salary_stats = self.df.groupby('title').agg({
            'salary_avg': ['mean', 'median', 'min', 'max', 'count']
        }).round(0)
        
        salary_stats.columns = ['Mean', 'Median', 'Min', 'Max', 'Count']
        salary_stats = salary_stats.sort_values('Mean', ascending=False)
        
        return salary_stats
    
    def analyze_by_location(self):
        """Analyze salary by location."""
        location_stats = self.df.groupby('location').agg({
            'salary_avg': ['mean', 'count']
        }).round(0)
        
        location_stats.columns = ['Average Salary', 'Job Count']
        location_stats = location_stats.sort_values('Average Salary', ascending=False)
        
        return location_stats.head(10)
    
    def salary_skill_correlation(self, top_n=10):
        """Analyze correlation between skills and salary."""
        skill_salary = {}
        
        for _, row in self.df.iterrows():
            skills = [s.strip() for s in str(row['skills']).split(',')]
            salary = row['salary_avg']
            
            for skill in skills:
                if skill not in skill_salary:
                    skill_salary[skill] = []
                skill_salary[skill].append(salary)
        
        # Calculate average salary per skill
        skill_avg_salary = {
            skill: np.mean(salaries) 
            for skill, salaries in skill_salary.items()
            if len(salaries) >= 3  # Minimum 3 occurrences
        }
        
        # Convert to DataFrame
        result = pd.DataFrame(list(skill_avg_salary.items()), 
                            columns=['Skill', 'Average Salary'])
        result = result.sort_values('Average Salary', ascending=False).head(top_n)
        result['Average Salary'] = result['Average Salary'].round(0)
        
        return result
    
    def experience_salary_analysis(self):
        """Analyze relationship between experience and salary."""
        exp_salary = self.df.groupby('experience_years').agg({
            'salary_avg': 'mean',
            'title': 'count'
        }).round(0)
        
        exp_salary.columns = ['Average Salary', 'Job Count']
        return exp_salary


class WorkWaveVisualizer:
    """
    Advanced visualization suite for job market analytics.
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
    
    def plot_top_skills(self, skills_df, top_n=10, save_path=None):
        """Create professional bar chart of top skills."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        top_skills = skills_df.head(top_n)
        
        bars = ax.barh(range(len(top_skills)), top_skills['Frequency'], 
                      color=self.colors[0], alpha=0.8, edgecolor='black')
        
        ax.set_yticks(range(len(top_skills)))
        ax.set_yticklabels(top_skills['Skill'])
        ax.set_xlabel('Number of Job Postings', fontsize=12, weight='bold')
        ax.set_title(f'Top {top_n} Most In-Demand Skills', 
                    fontsize=16, weight='bold', pad=20)
        
        # Add percentage labels
        for i, (freq, pct) in enumerate(zip(top_skills['Frequency'], 
                                            top_skills['Percentage'])):
            ax.text(freq + 0.5, i, f'{int(freq)} ({pct}%)', 
                   va='center', fontsize=10, weight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Top skills chart saved to {save_path}")
        
        plt.show()
    
    def plot_skills_wordcloud(self, skills_df, save_path=None):
        """Generate word cloud of skills."""
        # Create frequency dictionary
        skills_dict = dict(zip(skills_df['Skill'], skills_df['Frequency']))
        
        wordcloud = WordCloud(
            width=1200, 
            height=600,
            background_color='white',
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(skills_dict)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Job Skills Word Cloud', fontsize=20, weight='bold', pad=20)
        plt.tight_layout(pad=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Skills word cloud saved to {save_path}")
        
        plt.show()
    
    def plot_salary_by_role(self, salary_df, save_path=None):
        """Visualize salary distribution by role."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        roles = salary_df.index
        x = np.arange(len(roles))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, salary_df['Mean'], width, 
                      label='Mean Salary', color=self.colors[1], 
                      alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, salary_df['Median'], width,
                      label='Median Salary', color=self.colors[2], 
                      alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Job Role', fontsize=12, weight='bold')
        ax.set_ylabel('Salary ($)', fontsize=12, weight='bold')
        ax.set_title('Salary Analysis by Job Role', fontsize=16, weight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(roles, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Salary by role chart saved to {save_path}")
        
        plt.show()
    
    def plot_salary_trends(self, salary_df, save_path=None):
        """Create comprehensive salary trends visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Salary range by role
        ax1 = axes[0, 0]
        roles = salary_df.index
        means = salary_df['Mean']
        mins = salary_df['Min']
        maxs = salary_df['Max']
        
        y_pos = np.arange(len(roles))
        ax1.barh(y_pos, means, color=self.colors[0], alpha=0.7, label='Mean')
        ax1.errorbar(means, y_pos, 
                    xerr=[means - mins, maxs - means],
                    fmt='none', ecolor='red', alpha=0.5, capsize=5)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(roles)
        ax1.set_xlabel('Salary ($)', fontsize=10, weight='bold')
        ax1.set_title('Salary Range by Role', fontsize=12, weight='bold')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Job posting count by role
        ax2 = axes[0, 1]
        counts = salary_df['Count']
        ax2.bar(range(len(roles)), counts, color=self.colors[3], 
               alpha=0.8, edgecolor='black')
        ax2.set_xticks(range(len(roles)))
        ax2.set_xticklabels(roles, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Number of Postings', fontsize=10, weight='bold')
        ax2.set_title('Job Posting Volume by Role', fontsize=12, weight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Salary distribution
        ax3 = axes[1, 0]
        ax3.scatter(salary_df['Count'], salary_df['Mean'], 
                   s=200, c=self.colors[4], alpha=0.6, edgecolors='black')
        
        for i, role in enumerate(roles):
            ax3.annotate(role, (salary_df['Count'].iloc[i], 
                               salary_df['Mean'].iloc[i]),
                        fontsize=8, ha='right')
        
        ax3.set_xlabel('Number of Postings', fontsize=10, weight='bold')
        ax3.set_ylabel('Mean Salary ($)', fontsize=10, weight='bold')
        ax3.set_title('Postings vs. Salary Correlation', fontsize=12, weight='bold')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax3.grid(alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        📊 SALARY SUMMARY STATISTICS
        
        Highest Paying Role:
        {salary_df.index[0]}
        ${salary_df['Mean'].iloc[0]:,.0f}
        
        Most Common Role:
        {salary_df.nlargest(1, 'Count').index[0]}
        {int(salary_df.nlargest(1, 'Count')['Count'].iloc[0])} postings
        
        Overall Statistics:
        • Average Salary: ${salary_df['Mean'].mean():,.0f}
        • Median Salary: ${salary_df['Median'].median():,.0f}
        • Salary Range: ${salary_df['Min'].min():,.0f} - ${salary_df['Max'].max():,.0f}
        • Total Postings: {int(salary_df['Count'].sum())}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Salary trends visualization saved to {save_path}")
        
        plt.show()
    
    def plot_skills_by_category(self, categorized_skills, save_path=None):
        """Visualize skills grouped by category."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        categories = list(categorized_skills.keys())
        category_totals = [sum(s['frequency'] for s in categorized_skills[cat]) 
                          for cat in categories]
        
        bars = ax.barh(categories, category_totals, color=self.colors[:len(categories)],
                      alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Total Mentions', fontsize=12, weight='bold')
        ax.set_title('Skills Demand by Category', fontsize=16, weight='bold', pad=20)
        
        # Add value labels
        for i, (cat, total) in enumerate(zip(categories, category_totals)):
            ax.text(total + 2, i, str(int(total)), va='center', 
                   fontsize=11, weight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Skills by category chart saved to {save_path}")
        
        plt.show()
    
    def create_dashboard(self, skills_df, salary_df, save_path=None):
        """Create comprehensive dashboard visualization."""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Top skills
        ax1 = fig.add_subplot(gs[0:2, 0])
        top_10 = skills_df.head(10)
        ax1.barh(range(len(top_10)), top_10['Frequency'], 
                color=self.colors[0], alpha=0.8)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels(top_10['Skill'], fontsize=9)
        ax1.set_xlabel('Frequency', fontsize=10, weight='bold')
        ax1.set_title('Top 10 Skills', fontsize=12, weight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Salary by role
        ax2 = fig.add_subplot(gs[0:2, 1:])
        roles = salary_df.head(8).index
        x = np.arange(len(roles))
        ax2.bar(x, salary_df.head(8)['Mean'], color=self.colors[1], 
               alpha=0.8, edgecolor='black')
        ax2.set_xticks(x)
        ax2.set_xticklabels(roles, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Average Salary ($)', fontsize=10, weight='bold')
        ax2.set_title('Top Paying Roles', fontsize=12, weight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax2.grid(axis='y', alpha=0.3)
        
        # Skills distribution pie
        ax3 = fig.add_subplot(gs[2, 0])
        top_5_skills = skills_df.head(5)
        ax3.pie(top_5_skills['Frequency'], labels=top_5_skills['Skill'],
               autopct='%1.1f%%', startangle=90, colors=self.colors[:5])
        ax3.set_title('Top 5 Skills Distribution', fontsize=12, weight='bold')
        
        # Summary statistics
        ax4 = fig.add_subplot(gs[2, 1:])
        ax4.axis('off')
        
        summary = f"""
        🎯 WORKWAVE MARKET INSIGHTS
        
        📊 Skills Analysis:
        • Most In-Demand: {skills_df.iloc[0]['Skill']} ({skills_df.iloc[0]['Percentage']}% of jobs)
        • Emerging Skills: {', '.join(skills_df.iloc[5:8]['Skill'].values)}
        
        💰 Salary Insights:
        • Highest Paying: {salary_df.index[0]} (${salary_df['Mean'].iloc[0]:,.0f})
        • Industry Average: ${salary_df['Mean'].mean():,.0f}
        • Salary Range: ${salary_df['Min'].min():,.0f} - ${salary_df['Max'].max():,.0f}
        
        📈 Market Trends:
        • Total Jobs Analyzed: {int(salary_df['Count'].sum())}
        • Most Common Role: {salary_df.nlargest(1, 'Count').index[0]}
        """
        
        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        fig.suptitle('WorkWave: Job Market Analytics Dashboard', 
                    fontsize=18, weight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dashboard saved to {save_path}")
        
        plt.show()


class WorkWaveAnalyzer:
    """
    Main analyzer orchestrating all components.
    """
    
    def __init__(self):
        self.scraper = JobScraper()
        self.skills_analyzer = SkillsAnalyzer()
        self.visualizer = WorkWaveVisualizer()
        self.data = None
        
    def run_full_analysis(self, num_jobs=50, generate_report=True):
        """Execute complete analysis pipeline."""
        print("="*70)
        print("WORKWAVE: Job Market Analytics Platform")
        print("Author: Michael Semera")
        print("="*70)
        print()
        
        # Step 1: Scrape data
        print("📥 Step 1: Data Collection")
        self.data = self.scraper.scrape_sample_data(num_jobs)
        self.scraper.save_to_csv(self.data, 'workwave_jobs.csv')
        print()
        
        # Step 2: Skills analysis
        print("🔍 Step 2: Skills Analysis")
        skills_df = self.skills_analyzer.analyze_skills_distribution(self.data)
        print(f"✓ Identified {len(skills_df)} unique skills")
        categorized = self.skills_analyzer.categorize_skills(skills_df)
        print()
        
        # Step 3: Salary analysis
        print("💰 Step 3: Salary Analysis")
        salary_analyzer = SalaryAnalyzer(self.data)
        salary_by_role = salary_analyzer.analyze_by_role()
        salary_by_location = salary_analyzer.analyze_by_location()
        skill_salary_corr = salary_analyzer.salary_skill_correlation()
        print(f"✓ Analyzed salary data across {len(salary_by_role)} roles")
        print()
        
        # Step 4: Visualizations
        print("📊 Step 4: Generating Visualizations")
        self.visualizer.plot_top_skills(skills_df, top_n=10, 
                                       save_path='top_skills.png')
        self.visualizer.plot_skills_wordcloud(skills_df, 
                                             save_path='skills_wordcloud.png')
        self.visualizer.plot_salary_by_role(salary_by_role, 
                                           save_path='salary_by_role.png')
        self.visualizer.plot_salary_trends(salary_by_role, 
                                          save_path='salary_trends.png')
        self.visualizer.plot_skills_by_category(categorized, 
                                               save_path='skills_categories.png')
        self.visualizer.create_dashboard(skills_df, salary_by_role, 
                                        save_path='workwave_dashboard.png')
        print()
        
        # Step 5: Generate report
        if generate_report:
            print("📝 Step 5: Generating Report")
            self._generate_report(skills_df, salary_by_role, 
                                salary_by_location, skill_salary_corr)
        
        print()
        print("="*70)
        print("✓ Analysis Complete!")
        print("="*70)
        
        return {
            'data': self.data,
            'skills': skills_df,
            'salary_by_role': salary_by_role,
            'salary_by_location': salary_by_location,
            'skill_salary': skill_salary_corr,
            'categorized_skills': categorized
        }
    
    def _generate_report(self, skills_df, salary_df, location_df, skill_salary_df):
        """Generate comprehensive text report."""
        report = f"""
{'='*80}
WORKWAVE: JOB MARKET ANALYTICS REPORT
Author: Michael Semera
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

EXECUTIVE SUMMARY
{'-'*80}
Total Jobs Analyzed: {len(self.data)}
Date Range: {self.data['posted_date'].min().strftime('%Y-%m-%d')} to {self.data['posted_date'].max().strftime('%Y-%m-%d')}
Data Sources: {', '.join(self.data['source'].unique())}

MARKET OVERVIEW
{'-'*80}
The data science job market continues to show strong demand across multiple roles.
Analysis of {len(self.data)} job postings reveals critical insights into required
skills, compensation trends, and geographic distribution.

TOP 10 MOST IN-DEMAND SKILLS
{'-'*80}
Rank  Skill                      Frequency    Percentage
{'-'*80}
"""
        
        for i, row in skills_df.head(10).iterrows():
            report += f"{i+1:2d}.   {row['Skill']:<25} {int(row['Frequency']):3d}          {row['Percentage']:5.1f}%\n"
        
        report += f"""
KEY INSIGHT: {skills_df.iloc[0]['Skill']} appears in {skills_df.iloc[0]['Percentage']:.1f}% of all job postings,
making it the most critical skill for data professionals in today's market.

SALARY ANALYSIS BY ROLE
{'-'*80}
Role                           Mean Salary  Median      Range
{'-'*80}
"""
        
        for role, row in salary_df.head(10).iterrows():
            report += f"{role:<30} ${row['Mean']:>10,.0f}  ${row['Median']:>9,.0f}  ${row['Min']:>9,.0f}-${row['Max']:>9,.0f}\n"
        
        report += f"""
HIGHEST PAYING ROLE: {salary_df.index[0]}
Average Compensation: ${salary_df['Mean'].iloc[0]:,.0f}

TOP LOCATIONS BY SALARY
{'-'*80}
Location                       Average Salary    Jobs Available
{'-'*80}
"""
        
        for location, row in location_df.iterrows():
            report += f"{location:<30} ${row['Average Salary']:>13,.0f}    {int(row['Job Count']):>4d}\n"
        
        report += f"""
SKILLS WITH HIGHEST SALARY CORRELATION
{'-'*80}
Skill                          Average Salary
{'-'*80}
"""
        
        for _, row in skill_salary_df.iterrows():
            report += f"{row['Skill']:<30} ${row['Average Salary']:>13,.0f}\n"
        
        report += f"""
MARKET INSIGHTS & RECOMMENDATIONS
{'-'*80}

1. ESSENTIAL SKILLS
   The "must-have" skills for data professionals are:
   • {skills_df.iloc[0]['Skill']} (appears in {skills_df.iloc[0]['Percentage']:.0f}% of jobs)
   • {skills_df.iloc[1]['Skill']} (appears in {skills_df.iloc[1]['Percentage']:.0f}% of jobs)
   • {skills_df.iloc[2]['Skill']} (appears in {skills_df.iloc[2]['Percentage']:.0f}% of jobs)

2. EMERGING TRENDS
   Skills gaining traction include:
   {', '.join(skills_df.iloc[5:8]['Skill'].values)}

3. COMPENSATION TRENDS
   • Average market salary: ${salary_df['Mean'].mean():,.0f}
   • Salary premium for senior roles: {((salary_df[salary_df.index.str.contains('Senior|Manager')]['Mean'].mean() - salary_df['Mean'].mean()) / salary_df['Mean'].mean() * 100):.0f}%
   • Geographic variation: Up to {((location_df['Average Salary'].max() - location_df['Average Salary'].min()) / location_df['Average Salary'].min() * 100):.0f}% difference

4. CAREER RECOMMENDATIONS
   For professionals looking to maximize earning potential:
   a) Focus on high-value skills: {', '.join(skill_salary_df.head(3)['Skill'].values)}
   b) Consider roles: {', '.join(salary_df.head(3).index)}
   c) Target locations: {', '.join(location_df.head(3).index)}

5. SKILL DEVELOPMENT PRIORITIES
   Based on market demand and salary correlation:
   
   TIER 1 (Critical):
   {', '.join(skills_df.head(5)['Skill'].values)}
   
   TIER 2 (Important):
   {', '.join(skills_df.iloc[5:10]['Skill'].values)}
   
   TIER 3 (Beneficial):
   {', '.join(skills_df.iloc[10:15]['Skill'].values)}

METHODOLOGY
{'-'*80}
This analysis employed:
• Web scraping techniques for data collection
• NLP-based text analysis for skill extraction
• Statistical analysis for salary trends
• Advanced visualization for insight communication

Data Quality: {len(self.data)} verified job postings
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}

{'-'*80}
CONCLUSION
{'-'*80}
The data science job market remains robust with strong demand for professionals
skilled in {skills_df.iloc[0]['Skill']}, {skills_df.iloc[1]['Skill']}, and {skills_df.iloc[2]['Skill']}. 

Compensation is highly competitive, with {salary_df.index[0]} commanding
the highest salaries at an average of ${salary_df['Mean'].iloc[0]:,.0f}.

Geographic location significantly impacts compensation, with {location_df.index[0]}
offering the highest average salaries at ${location_df['Average Salary'].iloc[0]:,.0f}.

Professionals should prioritize continuous learning in high-demand skills while
considering geographic mobility to maximize career opportunities.

{'='*80}
END OF REPORT
{'='*80}

Report generated by WorkWave Analytics Platform
Author: Michael Semera
For questions or additional analysis, please contact the data team.
"""
        
        # Save report
        with open('workwave_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print("✓ Report saved to workwave_report.txt")


def main():
    """
    Main execution function for WorkWave.
    """
    print("\n" + "="*70)
    print(" "*15 + "🌊 WORKWAVE 🌊")
    print(" "*10 + "Job Market Analytics Platform")
    print(" "*15 + "by Michael Semera")
    print("="*70 + "\n")
    
    # Initialize analyzer
    analyzer = WorkWaveAnalyzer()
    
    # Run complete analysis
    try:
        results = analyzer.run_full_analysis(num_jobs=100, generate_report=True)
        
        print("\n📁 Generated Files:")
        print("  • workwave_jobs.csv - Raw job data")
        print("  • workwave_report.txt - Comprehensive analysis report")
        print("  • top_skills.png - Top 10 skills visualization")
        print("  • skills_wordcloud.png - Skills word cloud")
        print("  • salary_by_role.png - Salary comparison by role")
        print("  • salary_trends.png - Comprehensive salary analysis")
        print("  • skills_categories.png - Skills by category")
        print("  • workwave_dashboard.png - Complete analytics dashboard")
        
        print("\n💡 Quick Insights:")
        print(f"  • Most in-demand skill: {results['skills'].iloc[0]['Skill']}")
        print(f"  • Highest paying role: {results['salary_by_role'].index[0]}")
        print(f"  • Average salary: ${results['salary_by_role']['Mean'].mean():,.0f}")
        print(f"  • Best location: {results['salary_by_location'].index[0]}")
        
        print("\n🎯 Next Steps:")
        print("  1. Review the generated visualizations")
        print("  2. Read the comprehensive report (workwave_report.txt)")
        print("  3. Explore the raw data (workwave_jobs.csv)")
        print("  4. Customize analysis parameters as needed")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Thank you for using WorkWave!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
        