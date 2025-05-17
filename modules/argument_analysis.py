import spacy

class ArgumentAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_claims(self, text):
        """
        Extracts claims and subclaims from a given text.
        Args:
            text (str): Transcription text.
        Returns:
            dict: Claims and subclaims.
        """
        doc = self.nlp(text)
        claims = []
        subclaims = []

        for sent in doc.sents:
            root = [token for token in sent if token.dep_ == "ROOT"]
            if root:
                claims.append(sent.text)

        return {
            "claims": claims[:3],  # Limit to top 3 claims
            "subclaims": claims[3:]  # Treat the rest as subclaims
        }