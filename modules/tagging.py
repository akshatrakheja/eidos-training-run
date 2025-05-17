import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class ThemeTagger:
    def __init__(self, predefined_themes, model_name='all-MiniLM-L6-v2'):
        self.predefined_themes = predefined_themes
        self.model = SentenceTransformer(model_name)
        self.theme_embeddings = self.model.encode(predefined_themes)
        self.user_embeddings = []  # Store embeddings for clustering

    def tag_themes(self, text):
        """
        Tags the text with predefined themes and discovers dynamic themes.
        Args:
            text (str): Transcribed text from the audio.
        Returns:
            dict: Matched predefined themes and suggested new themes.
        """
        print("Tagging themes...")
        text_embedding = self.model.encode(text)
        self.user_embeddings.append(text_embedding)  # Store for clustering

        # Step 1: Predefined Theme Matching
        matched_themes = []
        for theme, theme_embedding in zip(self.predefined_themes, self.theme_embeddings):
            similarity = util.cos_sim(text_embedding, theme_embedding).item()
            print(f"Similarity with predefined theme '{theme}': {similarity}")  # Debugging
            if similarity > 0.3:  # Lower threshold for better matching
                matched_themes.append(theme)

        # Step 2: Dynamic Theme Discovery
        dynamic_themes = self.discover_dynamic_themes()

        return {"predefined_themes": matched_themes, "dynamic_themes": dynamic_themes}

    def discover_dynamic_themes(self, n_clusters=5):
        """
        Discovers dynamic themes using clustering or fallback to semantic summarization.
        Args:
            n_clusters (int): Number of clusters for KMeans.
        Returns:
            list: Suggested dynamic themes.
        """
        if len(self.user_embeddings) < 2:
            # Fallback: Semantic summarization with GPT
            print("Not enough embeddings for clustering. Using fallback.")
            text_snippets = [self.predefined_themes]
            return ["General Reflections", "Ethics of Gene Editing"]  # Example fallback themes

        # Clustering logic
        embeddings = np.array(self.user_embeddings)
        kmeans = KMeans(n_clusters=min(len(embeddings), n_clusters), random_state=42)
        kmeans.fit(embeddings)
        cluster_centroids = kmeans.cluster_centers_

        dynamic_themes = []
        for centroid in cluster_centroids:
            closest_theme_idx = util.cos_sim(centroid, self.theme_embeddings).argmax().item()
            dynamic_themes.append(self.predefined_themes[closest_theme_idx])

        return dynamic_themes