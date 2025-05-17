from pymongo import MongoClient

class StorageManager:
    def __init__(self, db_name="conversation_analysis"):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]

    def save_user_response(self, response):
        """
        Saves a user response to the database.
        Args:
            response (dict): Processed response with transcription and analysis.
        Returns:
            str: The ID of the inserted document.
        """
        collection = self.db["user_responses"]
        result = collection.insert_one(response)
        return str(result.inserted_id)

    def get_all_embeddings(self):
        """
        Retrieves all user embeddings for clustering.
        Returns:
            list: List of embeddings from all stored responses.
        """
        collection = self.db["user_responses"]
        embeddings = collection.find({}, {"embedding": 1, "_id": 0})
        return [doc["embedding"] for doc in embeddings]

    def save_cluster(self, cluster_data):
        """
        Saves cluster information to the database.
        Args:
            cluster_data (dict): Cluster metadata and associated inputs.
        Returns:
            str: The ID of the inserted document.
        """
        collection = self.db["clusters"]
        result = collection.insert_one(cluster_data)
        return str(result.inserted_id)