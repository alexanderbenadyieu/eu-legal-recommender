import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class EurLexEDA:
    def __init__(self, db_path='/data/eurlex.db'):
        """
        Comprehensive Exploratory Data Analysis for EurLex Dataset
        
        :param db_path: Path to the SQLite database
        """
        # Establish database connection
        self.conn = sqlite3.connect(db_path)
        
        # Load comprehensive dataset
        self.load_comprehensive_data()
        
        # Set up visualization style
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = (15, 10)
    
    def load_comprehensive_data(self):
        """
        Load a comprehensive dataset with multiple joins and aggregations
        """
        query = """
        SELECT 
            d.document_id, 
            d.celex_number, 
            d.title, 
            LENGTH(d.title) as title_length,
            d.eli_uri, 
            d.html_url,
            d.pdf_url,
            d.date_of_document,
            d.date_of_effect,
            d.date_of_end_validity,
            rb.body_name AS responsible_body,
            f.form_name AS document_form,
            GROUP_CONCAT(DISTINCT a.name) AS authors,
            GROUP_CONCAT(DISTINCT ed.descriptor_name) AS eurovoc_descriptors,
            GROUP_CONCAT(DISTINCT sm.subject_name) AS subject_matters,
            GROUP_CONCAT(DISTINCT dc.directory_code) AS directory_codes
        FROM 
            documents d
        LEFT JOIN responsible_bodies rb ON d.responsible_body_id = rb.responsible_body_id
        LEFT JOIN forms f ON d.form_id = f.form_id
        LEFT JOIN document_authors da ON d.document_id = da.document_id
        LEFT JOIN authors a ON da.author_id = a.author_id
        LEFT JOIN document_eurovoc_descriptors ded ON d.document_id = ded.document_id
        LEFT JOIN eurovoc_descriptors ed ON ded.descriptor_id = ed.descriptor_id
        LEFT JOIN document_subject_matters dsm ON d.document_id = dsm.document_id
        LEFT JOIN subject_matters sm ON dsm.subject_id = sm.subject_id
        LEFT JOIN document_directory_codes ddc ON d.document_id = ddc.document_id
        LEFT JOIN directory_codes dc ON ddc.directory_id = dc.directory_id
        GROUP BY 
            d.document_id
        """
        
        # Load data
        self.df = pd.read_sql_query(query, self.conn)
        
        # Data type conversions and preprocessing
        date_columns = ['date_of_document', 'date_of_effect', 'date_of_end_validity']
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Add derived columns
        self.df['year'] = self.df['date_of_document'].dt.year
        self.df['month'] = self.df['date_of_document'].dt.month
    
    def basic_dataset_overview(self):
        """
        Comprehensive dataset overview with multiple statistical insights
        """
        # Total documents and basic stats
        print("🔍 Dataset Overview:")
        print(f"Total Documents: {len(self.df)}")
        
        # Document form distribution
        print("\n📊 Document Form Distribution:")
        form_dist = self.df['document_form'].value_counts()
        print(form_dist)
        plt.figure(figsize=(12, 6))
        form_dist.plot(kind='bar')
        plt.title('Distribution of Document Forms')
        plt.xlabel('Document Form')
        plt.ylabel('Number of Documents')
        plt.tight_layout()
        plt.savefig('document_form_distribution.png')
        plt.close()
    
    def temporal_analysis(self):
        """
        In-depth temporal analysis of documents
        """
        print("\n⏰ Temporal Analysis:")
        
        # Yearly document count
        yearly_docs = self.df.groupby('year').size()
        print("\nDocuments per Year:")
        print(yearly_docs)
        
        plt.figure(figsize=(15, 6))
        yearly_docs.plot(kind='line', marker='o')
        plt.title('Number of Documents per Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Documents')
        plt.tight_layout()
        plt.savefig('documents_per_year.png')
        plt.close()
        
        # Monthly distribution
        monthly_docs = self.df.groupby('month').size()
        plt.figure(figsize=(12, 6))
        monthly_docs.plot(kind='bar')
        plt.title('Document Distribution by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Documents')
        plt.tight_layout()
        plt.savefig('documents_per_month.png')
        plt.close()
    
    def text_analysis(self):
        """
        Comprehensive text analysis
        """
        print("\n📝 Text Analysis:")
        
        # Title length analysis
        print("\nTitle Length Statistics:")
        print(self.df['title_length'].describe())
        
        plt.figure(figsize=(12, 6))
        self.df.boxplot(column='title_length', by='document_form')
        plt.title('Title Length by Document Form')
        plt.suptitle('')  # Remove automatic suptitle
        plt.tight_layout()
        plt.savefig('title_length_by_form.png')
        plt.close()
        
        # Word cloud of titles
        def preprocess_text(text):
            # Lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if w not in stop_words]
            return ' '.join(tokens)
        
        titles_text = ' '.join(self.df['title'].dropna())
        processed_titles = preprocess_text(titles_text)
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_titles)
        plt.figure(figsize=(16,8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Document Titles')
        plt.tight_layout()
        plt.savefig('titles_wordcloud.png')
        plt.close()
    
    def eurovoc_descriptors_analysis(self):
        """
        Comprehensive analysis of EuroVoc descriptors
        """
        print("\n🏷️ EuroVoc Descriptors Analysis:")
        
        # Explode descriptors
        descriptors = self.df['eurovoc_descriptors'].str.split(',', expand=True).stack()
        
        # Top descriptors
        top_descriptors = descriptors.value_counts().head(20)
        print("\nTop 20 EuroVoc Descriptors:")
        print(top_descriptors)
        
        plt.figure(figsize=(15, 7))
        top_descriptors.plot(kind='bar')
        plt.title('Top 20 EuroVoc Descriptors')
        plt.xlabel('Descriptor')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('top_eurovoc_descriptors.png')
        plt.close()
    
    def responsible_bodies_analysis(self):
        """
        Analysis of responsible bodies
        """
        print("\n🏢 Responsible Bodies Analysis:")
        
        # Top responsible bodies
        top_bodies = self.df['responsible_body'].value_counts().head(15)
        print("\nTop 15 Responsible Bodies:")
        print(top_bodies)
        
        plt.figure(figsize=(15, 7))
        top_bodies.plot(kind='bar')
        plt.title('Top 15 Responsible Bodies')
        plt.xlabel('Body')
        plt.ylabel('Number of Documents')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('top_responsible_bodies.png')
        plt.close()
    
    def comprehensive_analysis(self):
        """
        Run all analysis methods
        """
        self.basic_dataset_overview()
        self.temporal_analysis()
        self.text_analysis()
        self.eurovoc_descriptors_analysis()
        self.responsible_bodies_analysis()
    
    def __del__(self):
        """
        Close database connection
        """
        self.conn.close()

def main():
    eda = EurLexEDA()
    eda.comprehensive_analysis()

if __name__ == '__main__':
    main()