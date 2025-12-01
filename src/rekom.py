import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CollaborativeFilteringRecommender:
    def __init__(self, data_path, num_users=500, random_seed=42):
        self.data_path = data_path
        self.num_users = num_users
        self.random_seed = random_seed
        self.df = None
        self.user_item_matrix = None
        self.item_similarity_df = None
        self.user_ids = None
        self.interactions_df = None
        self._load_and_preprocess_data()
        self._simulate_interactions()
        self._build_similarity_matrix()
        logger.info("CF initialized.")

    def _load_and_preprocess_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        required_cols = ['Rating','ReviewCount','Description','Tags','Name','Category','Brand','ProdID','ImageURL']
        for c in required_cols:
            if c not in self.df.columns:
                self.df[c] = '' if c in ['Description','Tags','Name','Category','Brand','ImageURL'] else 0
        self.df['Rating'] = pd.to_numeric(self.df['Rating'], errors='coerce')
        self.df['ReviewCount'] = pd.to_numeric(self.df['ReviewCount'], errors='coerce').fillna(0)
        # Rating 0 → NaN → diisi mean per item → minimal 1
        self.df.loc[self.df['Rating'] == 0, 'Rating'] = np.nan
        self.df['Rating'] = self.df.groupby('ProdID')['Rating'].transform(
            lambda x: x.fillna(x.mean() if not np.isnan(x.mean()) else 1)
        )
        # Pastikan tidak ada rating 0
        self.df['Rating'] = self.df['Rating'].clip(lower=1)
        self.df['Description'] = self.df['Description'].fillna('')
        self.df['Name'] = self.df['Name'].fillna('')
        self.df['Category'] = self.df['Category'].fillna('Unknown')
        if 'ProdID' not in self.df.columns:
            self.df['ProdID'] = ['prod_' + str(i) for i in range(len(self.df))]
        self.df = self.df.drop_duplicates(subset=['ProdID']).reset_index(drop=True)
        # Buat harga dummy
        np.random.seed(self.random_seed)
        self.df['Price'] = np.random.randint(50000, 500000, size=len(self.df))
        logger.info(f"Dataset loaded: {self.df.shape}")

    def _simulate_interactions(self):
        np.random.seed(self.random_seed)
        self.user_ids = [f"user_{i}" for i in range(self.num_users)]
        interactions = []
        for _, row in self.df.iterrows():
            prod_id = row['ProdID']
            base_rating = float(row['Rating'])
            num_interact = np.random.randint(1, 20)
            for _ in range(num_interact):
                user = np.random.choice(self.user_ids)
                rating = np.clip(np.random.normal(base_rating, 0.4), 1, 5)
                interactions.append({'user_id': user, 'prod_id': prod_id, 'rating': round(float(rating),1)})
        self.interactions_df = pd.DataFrame(interactions)
        logger.info(f"Simulated interactions: {len(self.interactions_df)} rows")

    def _build_similarity_matrix(self):
        self.user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', columns='prod_id', values='rating', fill_value=0
        )
        item_sim = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df = pd.DataFrame(
            item_sim, index=self.user_item_matrix.columns, columns=self.user_item_matrix.columns
        )
        logger.info("Similarity matrix built.")

    def get_most_liked_products(self, top_n=10):
        if self.interactions_df is None or self.interactions_df.empty:
            logger.warning("No interactions data available to determine most liked products.")
            return pd.DataFrame()
        # Hitung rata-rata rating untuk setiap produk
        product_avg_ratings = self.interactions_df.groupby('prod_id')['rating'].mean().reset_index()
        product_avg_ratings = product_avg_ratings.rename(columns={'rating': 'average_rating', 'prod_id': 'ProdID'})
        # Gabungkan dengan dataframe produk asli untuk mendapatkan detail lainnya
        most_liked = pd.merge(product_avg_ratings, self.df, on='ProdID', how='left')
        # Urutkan berdasarkan rata-rata rating tertinggi dan ambil top_n
        most_liked = most_liked.sort_values(by='average_rating', ascending=False)
        return most_liked.head(top_n)[['ProdID', 'Name', 'Brand', 'Category', 'average_rating', 'ReviewCount', 'ImageURL', 'Description', 'Price']]