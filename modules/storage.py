import json
import os

class StorageManager:
    def __init__(self, local_dir="local_storage"):
        """
        Initializes local storage for saving user responses and clusters.
        Args:
            local_dir (str): Directory for local storage files.
        """
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)

    def save_user_response(self, response):
        """
        Saves a user response locally to a JSON file.
        Args:
            response (dict): Processed response with transcription and analysis.
        Returns:
            str: Path to the saved JSON file.
        """
        file_path = os.path.join(self.local_dir, f"user_response_{response['user_id']}.json")
        with open(file_path, "w") as f:
            json.dump(response, f, indent=4)
        return file_path

    def get_all_embeddings(self):
        """
        Retrieves all user embeddings from local JSON files.
        Returns:
            list: List of embeddings from all stored responses.
        """
        embeddings = []
        for file_name in os.listdir(self.local_dir):
            if file_name.startswith("user_response_") and file_name.endswith(".json"):
                with open(os.path.join(self.local_dir, file_name), "r") as f:
                    response = json.load(f)
                    embeddings.append(response["embedding"])
        return embeddings

    def save_cluster(self, cluster_data):
        """
        Saves cluster information locally to a JSON file.
        Args:
            cluster_data (dict): Cluster metadata and associated inputs.
        Returns:
            str: Path to the saved JSON file.
        """
        file_path = os.path.join(self.local_dir, f"cluster_{cluster_data['cluster_id']}.json")
        with open(file_path, "w") as f:
            json.dump(cluster_data, f, indent=4)
        return file_path