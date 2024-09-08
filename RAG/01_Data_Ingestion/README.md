## Data Chunking

``` mermaid
flowchart TB
    Document --> RecursiveCharacterTextSplitter & TokenTextSplitter & KamradtSemanticChunker & KamradtModifiedChunker & ClusterSemanticChunker & LLMSemanticChunker

    subgraph "1. RecursiveCharacterTextSplitter"
        RecursiveCharacterTextSplitter --> RCS1[Split by separators]
        RCS1 --> RCS2["Priority: <br/>\n\n, \n, ., ?, !, space"]
        RCS2 --> RCS3[Merge splits until max length]
        RCS3 --> RCS4[Optional: Add chunk overlap]
    end

    subgraph "2. TokenTextSplitter"
        TokenTextSplitter --> TTS1[Tokenize text]
        TTS1 --> TTS2[Split by fixed token count]
        TTS2 --> TTS3[Ensure splits at token boundaries]
        TTS3 --> TTS4[Optional: Add chunk overlap]
    end

    subgraph "3. KamradtSemanticChunker"
        KamradtSemanticChunker --> KSC1[Split by sentence]
        KSC1 --> KSC2[Compute embeddings<br/>for sliding window]
        KSC2 --> KSC3[Calculate cosine distances<br/>between consecutive windows]
        KSC3 --> KSC4[Find discontinuities<br/>> 95th percentile]
        KSC4 --> KSC5[Split at discontinuities]
    end

    subgraph "4. KamradtModifiedChunker"
        KamradtModifiedChunker --> KMC1[Split by sentence]
        KMC1 --> KMC2[Compute embeddings<br/>for sliding window]
        KMC2 --> KMC3[Calculate cosine distances<br/>between consecutive windows]
        KMC3 --> KMC4[Binary search for<br/>optimal threshold]
        KMC4 --> KMC5[Ensure largest chunk<br/>< specified length]
        KMC5 --> KMC6[Split at determined<br/>discontinuities]
    end

    subgraph "5. ClusterSemanticChunker"
        ClusterSemanticChunker --> CSC1[Split into 50-token pieces]
        CSC1 --> CSC2[Compute embeddings<br/>for each piece]
        CSC2 --> CSC3[Calculate pairwise<br/>cosine similarities]
        CSC3 --> CSC4[Use dynamic programming<br/>to maximize similarity]
        CSC4 --> CSC5[Ensure chunks <= max length]
        CSC5 --> CSC6[Merge pieces into<br/>optimal chunks]
    end

    subgraph "6. LLMSemanticChunker"
        LLMSemanticChunker --> LSC1[Split into 50-token pieces]
        LSC1 --> LSC2[Surround with<br/><start_chunk_X> tags]
        LSC2 --> LSC3[Prompt LLM with tagged text]
        LSC3 --> LSC4[LLM returns split indexes]
        LSC4 --> LSC5[Process indexes to<br/>create chunks]
        LSC5 --> LSC6[Ensure chunks <= max length]
    end

    %% Force diagram to render left-to-right
    RecursiveCharacterTextSplitter ~~~ TokenTextSplitter ~~~ KamradtSemanticChunker ~~~ KamradtModifiedChunker ~~~ ClusterSemanticChunker ~~~ LLMSemanticChunker
```