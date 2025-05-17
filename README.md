# Conversation Analysis RAG System

A Retrieval-Augmented Generation (RAG) system for analyzing, generating, and improving conversational data.

## Overview

This repository implements a system designed to facilitate interactive conversations, collect user feedback, and use that data to fine-tune language models. The system uses a RAG approach to retrieve and generate relevant responses based on past conversations and real-world data.

## Key Components

1. **RAG Pipelines**:
   - `ragpipeline.py` - Core pipeline for embedding, retrieval, and question generation
   - `ragpipeline2.py` - Enhanced pipeline with tone/intensity analysis and audio features

2. **Data Processing Modules**:
   - Various utilities for processing conversation data, real-world data, and user feedback

3. **Vector Database Integration**:
   - Uses Pinecone for storing and retrieving embeddings

4. **Fine-tuning Pipeline**:
   - Collects conversation data with user ratings for model improvement

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys and configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   INDEX_NAME=eidos-data
   DIMENSION=1536
   METRIC=cosine
   NAMESPACE_SENTENCES=sentences
   NAMESPACE_CONVERSATIONS=conversations
   OUTPUT_FOLDER=newconversations
   ```
4. Run the pipeline:
   ```bash
   python pipelines/ragpipeline2.py
   ```

## Architecture

The system follows this general workflow:
1. Process and embed conversational data
2. Store embeddings in a vector database
3. Retrieve relevant responses based on user voice input
4. Generate follow-up questions using fine-tuned LLM
5. Collect user feedback for continuous improvement

## Modules

- `clean.py` - Fixes role errors in conversation data
- `processtranscriptforugc.py` - Processes transcripts with emotion classification
- `interaction_manager.py` - Manages user interactions
- `feedback.py` - Collects and analyzes user feedback
- `loadrealworld.py` - Extracts real-world data for context
- And more...

## License