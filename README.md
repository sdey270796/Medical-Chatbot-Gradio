# Medical Chatbot with LLM --- Fibromyalgia RAG Assistant

> **Retrieval-Augmented Generation (RAG) prototype for medical
> information retrieval and conversational question answering**

This project implements a **medical-domain conversational chatbot**
using a local Large Language Model (LLM), semantic embeddings, vector
search, and LangChain.

The prototype is built around a single authoritative knowledge source: a
**Fibromyalgia patient-information booklet from Versus Arthritis**. The
notebook demonstrates the complete pipeline from PDF ingestion and
chunking to embeddings, FAISS retrieval, conversational question
reformulation, LLM generation, and a Gradio interface.

The project is particularly useful as a practical demonstration of how
**Generative AI + RAG + local LLM inference** can be combined to build a
domain-specific question-answering system.

------------------------------------------------------------------------

## 📌 Project Overview

Large Language Models can generate fluent answers, but a general-purpose
model may not reliably contain the precise information required for a
specialized domain.

A Retrieval-Augmented Generation system addresses this problem by
introducing an external knowledge source:

``` text
                  User Question
                       │
                       ▼
               Query Representation
                       │
                       ▼
              Semantic Retrieval
                       │
                       ▼
             Relevant Document Chunks
                       │
                       ▼
                  LLM Context
                       │
                       ▼
                 Generated Answer
```

In this project, the external knowledge source is a **Fibromyalgia
information booklet**.

The notebook implements:

-   Local LLaMA-based text generation
-   PDF document loading
-   Recursive text chunking
-   Sentence-transformer embeddings
-   FAISS vector similarity search
-   LangChain retrievers
-   Conversational retrieval
-   Follow-up-question condensation
-   Gradio deployment
-   GPU-assisted embedding generation

The source document is a patient-information booklet covering
fibromyalgia, including symptoms, diagnosis, treatment, self-management,
research, and related information. The supplied PDF contains 17 PDF
pages and its internal contents extend across sections such as
diagnosis, treatments, self-management, research, glossary, and further
information. fileciteturn18file0L12-L14
fileciteturn18file0L32-L51

------------------------------------------------------------------------

# 🎯 Objectives

The primary objectives of the project are to:

1.  Build a domain-specific medical question-answering system.
2.  Use an external medical-information document as the knowledge
    source.
3.  Convert unstructured PDF content into searchable semantic
    representations.
4.  Implement vector similarity search using FAISS.
5.  Use a local LLaMA model instead of relying on a hosted LLM API.
6.  Construct a conversational retrieval chain using LangChain.
7.  Handle follow-up questions using chat-history-aware question
    reformulation.
8.  Demonstrate a working natural-language interface through Gradio.
9.  Explore the architecture and practical requirements of a small RAG
    system.
10. Highlight the difference between a basic LLM chatbot and a
    retrieval-grounded chatbot.

------------------------------------------------------------------------

# 🧠 Why RAG?

A conventional LLM interaction looks like:

``` text
Question
   │
   ▼
LLM
   │
   ▼
Answer
```

The model is responsible for generating the response from its learned
parameters.

A RAG system instead introduces an explicit knowledge-retrieval stage:

``` text
Question
   │
   ▼
Embedding
   │
   ▼
Vector Search
   │
   ▼
Relevant Documents
   │
   ▼
LLM
   │
   ▼
Answer
```

This allows the application to use a specific document collection as its
knowledge base.

For this project, that collection is intentionally narrow: the supplied
**Fibromyalgia** booklet.

------------------------------------------------------------------------

# 📚 Knowledge Source

The repository's second file is the project's knowledge source:

``` text
fibromyalgia-information-booklet-july2021.pdf
```

The document is a patient-oriented publication from **Versus
Arthritis**.

The booklet explains:

-   What fibromyalgia is
-   Common symptoms
-   Who may develop it
-   Possible causes and contributing factors
-   Diagnosis
-   Treatment approaches
-   Physical therapies
-   Pain clinics and pain-management programmes
-   Psychological therapies
-   Self-management
-   Research and new developments
-   Glossary and additional resources

The contents page of the supplied document explicitly lists these
sections. fileciteturn18file0L32-L51

The document also emphasizes that experiences and responses to treatment
can vary between individuals and presents its information as a
combination of experiences, research, and facts.
fileciteturn18file0L16-L30

------------------------------------------------------------------------

# 🏥 Medical Safety Scope

This project should be treated as an **educational RAG prototype**, not
a clinical decision-support system.

It is not designed to:

-   Diagnose diseases
-   Replace a doctor or other qualified healthcare professional
-   Prescribe medication
-   Determine emergency treatment
-   Provide personalized medical decisions
-   Guarantee factual or clinically appropriate answers

The underlying source itself is patient-information material rather than
a substitute for professional medical care.

A production medical chatbot would require substantially stronger
safeguards, including:

-   Clinical validation
-   Source citation
-   Hallucination detection
-   Retrieval-quality evaluation
-   Medical-domain evaluation datasets
-   Safety classifiers
-   Refusal mechanisms
-   Human oversight
-   Versioned medical sources
-   Regulatory and privacy considerations

------------------------------------------------------------------------

# 🏗️ System Architecture

The notebook contains two related implementations.

## Conceptual RAG Pipeline

``` text
┌─────────────────────────────┐
│ Fibromyalgia PDF            │
│ Knowledge Source             │
└──────────────┬──────────────┘
               │
               ▼
        PDF Document Loader
               │
               ▼
        Recursive Chunking
               │
               ▼
       Sentence Embeddings
               │
               ▼
         FAISS Vector DB
               │
               ▼
            Retriever
               │
               ▼
         Relevant Chunks
               │
               ▼
       Conversational RAG
               │
               ▼
         Local LLaMA 2
               │
               ▼
             Answer
```

The notebook implements the retrieval chain with LangChain's
`ConversationalRetrievalChain`, configured with a retriever and a prompt
for reformulating follow-up questions. fileciteturn10file0L605-L678

------------------------------------------------------------------------

# 🧰 Technology Stack

  Component                   Technology
  --------------------------- ---------------------------------------------
  Programming language        Python
  Notebook environment        Google Colab
  LLM                         LLaMA 2 7B Chat
  LLM runtime                 CTransformers
  Framework                   LangChain
  Embeddings                  BAAI/bge-base-en-v1.5
  Vector database             FAISS
  PDF loader                  PyPDFLoader
  Text splitting              RecursiveCharacterTextSplitter
  Secondary embedding model   all-MiniLM-L6-v2
  Interface                   Gradio
  GPU                         NVIDIA T4 in the recorded Colab environment
  Model download              Google Drive via `gdown`

The notebook metadata indicates a Colab GPU configuration using a T4.
fileciteturn8file0L13-L27

------------------------------------------------------------------------

# 🤖 Local LLM

One of the notable aspects of the project is that the chatbot does **not
rely on a hosted LLM API**.

The notebook downloads and loads:

``` text
llama-2-7b-chat.ggmlv3.q4_0.bin
```

The recorded download is approximately:

``` text
3.79 GB
```

and the model is loaded through:

``` python
CTransformers
```

with:

``` python
model_type = "llama"
```

The recorded configuration is:

``` python
{
    "max_new_tokens": 600,
    "temperature": 0.01,
    "context_length": 5000
}
```

The low temperature is intended to make generation more deterministic
and less creative---an appropriate direction for an
information-retrieval-oriented application.

The notebook explicitly downloads both the Fibromyalgia PDF and the
LLaMA model into a Google Colab `Models/GenAI Assignment` directory.
fileciteturn10file0L23-L104

------------------------------------------------------------------------

# 🔤 Embedding Model

The principal embedding model used for the RAG pipeline is:

``` text
BAAI/bge-base-en-v1.5
```

It is loaded using:

``` python
SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)
```

The notebook subsequently integrates the model with LangChain through:

``` python
HuggingFaceEmbeddings
```

and configures it to use:

``` python
device = "cuda"
```

with normalized embeddings:

``` python
normalize_embeddings = True
```

This makes the embedding stage suitable for semantic similarity search
in the vector database. fileciteturn10file0L135-L147
fileciteturn10file0L451-L490

------------------------------------------------------------------------

# 📄 PDF Ingestion

The document is loaded with:

``` python
from langchain.document_loaders import PyPDFLoader

pdf_reader = PyPDFLoader(
    "/content/Models/GenAI Assignment/fibromyalgia-information-booklet-july2021.pdf"
)

document = pdf_reader.load()
```

The notebook therefore converts the PDF into a collection of LangChain
document objects before further processing.
fileciteturn10file0L497-L538

------------------------------------------------------------------------

# ✂️ Document Chunking

Large documents cannot simply be inserted into an LLM context window in
their entirety.

The notebook therefore splits the PDF into smaller chunks using:

``` python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
```

The resulting workflow is:

``` text
PDF
 │
 ▼
Pages / Documents
 │
 ▼
Recursive Character Splitting
 │
 ├── Chunk 1
 ├── Chunk 2
 ├── Chunk 3
 ├── ...
 └── Chunk N
```

### Chunk Configuration

  Parameter                   Value
  --------------- -----------------
  Chunk size        1000 characters
  Chunk overlap      200 characters

The overlap helps preserve contextual continuity between adjacent
chunks.

The notebook implements this directly using
`RecursiveCharacterTextSplitter`. fileciteturn10file0L540-L569

------------------------------------------------------------------------

# 🗃️ FAISS Vector Database

After chunking, the document chunks are embedded and inserted into a
FAISS vector store.

The notebook uses:

``` python
from langchain.vectorstores import FAISS

vector_db = FAISS.from_documents(
    documents=chunk,
    embedding=embeddings
)
```

This transforms the document collection into a semantic search system.

Conceptually:

``` text
Text Chunk
    │
    ▼
Embedding Model
    │
    ▼
Vector
    │
    ▼
FAISS
```

A user query can then be represented in the same semantic space and
compared against the stored document vectors.

The FAISS vector store and document-embedding operation are explicitly
implemented in the notebook. fileciteturn10file0L573-L603

------------------------------------------------------------------------

# 🔎 Retriever

The FAISS database is converted into a LangChain retriever:

``` python
retriever = vector_db.as_retriever()
```

The retriever provides the bridge between:

``` text
User Question
      │
      ▼
Semantic Search
      │
      ▼
Relevant PDF Chunks
```

This is the retrieval component of the RAG architecture.
fileciteturn10file0L605-L623

------------------------------------------------------------------------

# 💬 Conversational Retrieval

A particularly useful component of the notebook is its treatment of
conversational questions.

The project defines a prompt that asks the model to transform a
follow-up question into a standalone question using the previous
conversation:

``` text
Chat History
     +
Follow-up Question
     ↓
Standalone Question
```

The prompt structure is:

``` text
Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question.
```

This is important because a user may ask:

``` text
What are the symptoms?

How long do they last?

What about treatment?
```

The second and third questions depend on conversational context.

A retrieval system needs a more complete standalone representation of
the user's intent before performing semantic search.

The notebook implements this using:

``` python
PromptTemplate
```

and:

``` python
ConversationalRetrievalChain
```

with:

``` python
return_source_documents=True
```

The recorded implementation therefore contains the core components
expected in a conversational RAG pipeline.
fileciteturn10file0L625-L684

------------------------------------------------------------------------

# 🧠 RAG Inference

The notebook tests the conversational retrieval chain using:

``` python
query = "What are the main symptoms of fibromyalgia?"

result = chain.invoke(
    {
        "question": query,
        "chat_history": chat_history
    }
)

print(result["answer"])
```

The recorded response states that major symptoms include widespread pain
together with tiredness, fatigue, and lack of energy.
fileciteturn14file0L1-L15

The answer is broadly consistent with the supplied source document,
which discusses widespread pain, fatigue, sleep disturbance, and
cognitive difficulties among the symptoms and related features of
fibromyalgia. fileciteturn7file7L421-L454

------------------------------------------------------------------------

# 🗣️ Conversation Memory

The notebook demonstrates manual maintenance of conversational history
using:

``` python
HumanMessage
AIMessage
```

The first question-answer pair is added to:

``` python
chat_history
```

so that subsequent questions can be interpreted in context.

The recorded history contains:

``` text
Human:
What are the main symptoms of fibromyalgia?

AI:
The main symptoms ...
```

This demonstrates the basic mechanism required for multi-turn
retrieval-based conversations. fileciteturn14file2L20-L45

------------------------------------------------------------------------

# 🖥️ Gradio Deployment

The project also attempts to expose the chatbot through a lightweight
web interface using **Gradio**.

The interface is created as:

``` python
interface = gr.Interface(
    fn=rag_inference,
    inputs="text",
    outputs="text",
    title="Medical Chatbot",
    description="Ask a question, and the model will generate an answer using RAG."
)
```

The notebook then launches it with:

``` python
interface.launch()
```

The recorded Colab execution generated a temporary public Gradio URL,
demonstrating that the interface was successfully launched in the
notebook environment. fileciteturn13file0L115-L171

------------------------------------------------------------------------

# ⚠️ Important Implementation Note

There is an important distinction between the **working conversational
RAG chain** and the **final Gradio inference function** in the notebook.

The earlier pipeline correctly constructs:

``` text
PDF
 → Chunks
 → BGE Embeddings
 → FAISS
 → Retriever
 → ConversationalRetrievalChain
 → LLaMA
```

However, the final Gradio function is implemented as:

``` python
def rag_inference(query):
    query_embedding = retriever_model.encode(query)
    response = llm(query)
    return response
```

The calculated:

``` python
query_embedding
```

is not subsequently used for retrieval, and the `vector_db`/`retriever`
is not invoked inside this function. fileciteturn13file0L87-L112

Therefore, **as written, the deployed Gradio interface does not actually
perform the same RAG retrieval step demonstrated earlier in the
notebook**.

This is an important technical observation and is intentionally
documented here rather than describing the final interface as a fully
retrieval-grounded chatbot.

### Actual final deployment path

``` text
User Query
    │
    ▼
all-MiniLM-L6-v2
    │
    ▼
Query Embedding
    │
    │   [Embedding is not used]
    │
    ▼
LLaMA 2
    │
    ▼
Generated Answer
```

The LLM therefore receives the raw query directly.

------------------------------------------------------------------------

# 🔄 Two Embedding Models in the Notebook

Another noteworthy implementation detail is that the notebook uses two
different embedding models at different stages.

### RAG construction

``` text
BAAI/bge-base-en-v1.5
```

is used with LangChain's `HuggingFaceEmbeddings` to construct the FAISS
vector database.

### Gradio deployment

``` text
all-MiniLM-L6-v2
```

is loaded as:

``` python
retriever_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
```

The second model is used to encode the user query in `rag_inference`,
but the resulting vector is not passed to the FAISS retriever.
fileciteturn10file0L1047-L1069 fileciteturn13file0L87-L112

For a production implementation, the embedding model used for query
encoding should normally be aligned with the embedding space used to
build the vector database.

------------------------------------------------------------------------

# 📊 Source Document Coverage

The supplied Fibromyalgia booklet provides a reasonably broad knowledge
base for a focused educational chatbot.

Its contents include:

  Topic                     Included
  ------------------------- ----------
  What is fibromyalgia?     ✓
  Symptoms                  ✓
  Who gets it?              ✓
  Family history            ✓
  Causes                    ✓
  Diagnosis                 ✓
  Prognosis / future        ✓
  Treatments                ✓
  Physiotherapy             ✓
  Occupational therapy      ✓
  Pain management           ✓
  Psychological therapies   ✓
  Drug treatments           ✓
  Self-management           ✓
  Exercise                  ✓
  Sleep                     ✓
  Diet and nutrition        ✓
  Research                  ✓
  Glossary                  ✓
  Further information       ✓

The source's contents page explicitly lists the major sections through
"Where can I find out more?" and "Talk to us."
fileciteturn18file0L32-L51

------------------------------------------------------------------------

# 🔬 Example Knowledge Areas

The source document covers clinically relevant educational topics such
as:

### Symptoms

The booklet discusses widespread pain, fatigue, unrefreshing sleep,
memory/concentration problems, and the interaction between pain, sleep,
stress, and mood. fileciteturn7file7L421-L454

### Diagnosis

It explains that diagnosis can be difficult because symptoms vary
between individuals and that clinicians may investigate other conditions
with similar symptoms. fileciteturn7file8L505-L539

### Treatment

The source discusses physical therapies, psychological approaches,
medication, pain-management programmes, and self-management.
fileciteturn7file9L595-L617

### Self-Management

The booklet includes guidance around pacing activities, sleep, exercise,
stress, workplace adaptations, relaxation, and other supportive
strategies. fileciteturn7file2L92-L130

### Research

The source also describes research into pain processing, stress
responses, treatment matching, and healthcare services.
fileciteturn7file5L298-L328

------------------------------------------------------------------------

# 📦 Installation

The notebook installs a range of packages including:

``` bash
pip install langchain
pip install langchain_community
pip install sentence_transformers
pip install bitsandbytes
pip install accelerate
pip install ctransformers
pip install pypdf
pip install faiss-gpu
```

For the Gradio stage it installs:

``` bash
pip install gradio langchain sentence-transformers ctransformers
```

The recorded environment used versions including:

``` text
LangChain            0.3.1
Sentence Transformers 3.1.1
CTransformers        0.2.27
Gradio               4.44.1
PyTorch              2.4.1+cu121
Transformers         4.44.2
Python               3.10
```

These versions describe the recorded notebook environment and should not
be interpreted as a guaranteed current compatibility matrix.

------------------------------------------------------------------------

# 🚀 Running the Project

The notebook was developed for **Google Colab**.

## Step 1 --- Open the Notebook

Open:

``` text
Medical_Chatbot_with_LLM.ipynb
```

in Google Colab.

A GPU runtime is recommended because the notebook uses a local LLaMA
model and GPU-backed embeddings.

------------------------------------------------------------------------

## Step 2 --- Install Dependencies

Run the package-installation cells.

------------------------------------------------------------------------

## Step 3 --- Obtain the Model and Source Document

The notebook uses `gdown` to download:

``` text
fibromyalgia-information-booklet-july2021.pdf
```

and:

``` text
llama-2-7b-chat.ggmlv3.q4_0.bin
```

from a Google Drive folder.

The recorded notebook downloads approximately 2.22 MB of PDF data and
approximately 3.79 GB of model data. fileciteturn10file0L63-L92

------------------------------------------------------------------------

## Step 4 --- Initialize the LLM

Load the LLaMA model through CTransformers.

------------------------------------------------------------------------

## Step 5 --- Create Embeddings

Load:

``` text
BAAI/bge-base-en-v1.5
```

and configure GPU execution.

------------------------------------------------------------------------

## Step 6 --- Load the PDF

Use:

``` python
PyPDFLoader
```

to parse the source document.

------------------------------------------------------------------------

## Step 7 --- Split the Document

Use:

``` python
RecursiveCharacterTextSplitter
```

with:

``` text
chunk_size = 1000
chunk_overlap = 200
```

------------------------------------------------------------------------

## Step 8 --- Build FAISS

Create the vector store:

``` python
FAISS.from_documents(...)
```

------------------------------------------------------------------------

## Step 9 --- Create Retriever

Convert the vector store into a retriever:

``` python
vector_db.as_retriever()
```

------------------------------------------------------------------------

## Step 10 --- Build Conversational Retrieval Chain

Configure:

``` python
ConversationalRetrievalChain.from_llm(...)
```

with the local LLaMA model and retriever.

------------------------------------------------------------------------

## Step 11 --- Test Questions

For example:

``` text
What are the main symptoms of fibromyalgia?
```

The notebook demonstrates this query and produces an answer.
fileciteturn14file0L4-L15

------------------------------------------------------------------------

## Step 12 --- Launch Gradio

Run:

``` python
interface.launch()
```

to expose the prototype interface.

------------------------------------------------------------------------

# 📁 Repository Structure

With the two files provided for this project, a clean repository
structure is:

``` text
.
├── README.md
├── Medical_Chatbot_with_LLM.ipynb
└── fibromyalgia-information-booklet-july2021.pdf
```

The two files have distinct roles:

``` text
Medical_Chatbot_with_LLM.ipynb
        │
        └── Implementation

fibromyalgia-information-booklet-july2021.pdf
        │
        └── Knowledge Base
```

The original LLaMA model is **not recommended for inclusion in a Git
repository** because the recorded model file is approximately 3.79 GB.
The notebook instead downloads it externally using `gdown`.
fileciteturn10file0L63-L92

------------------------------------------------------------------------

# 🧮 End-to-End Data Flow

## Knowledge-Base Construction

``` text
Fibromyalgia PDF
       │
       ▼
    PyPDFLoader
       │
       ▼
Document Objects
       │
       ▼
RecursiveCharacterTextSplitter
       │
       ▼
1000-character Chunks
       │
       ▼
BGE Embeddings
       │
       ▼
FAISS Vector Database
```

## Conversational Query

``` text
User Question
       │
       ▼
Conversation History
       │
       ▼
Question Condensation
       │
       ▼
Semantic Retrieval
       │
       ▼
Relevant Document Chunks
       │
       ▼
LLaMA 2 7B
       │
       ▼
Answer
```

This is the intended RAG workflow implemented through the notebook's
LangChain retrieval chain. fileciteturn10file0L625-L684

------------------------------------------------------------------------

# 🧪 What This Project Demonstrates

This project provides a compact demonstration of several important
Generative AI concepts.

### 1. Domain-Specific LLM Applications

A general LLM can be wrapped around a narrow domain-specific knowledge
source.

### 2. Semantic Search

Documents can be searched according to meaning rather than simple
keyword matching.

### 3. Vector Databases

FAISS provides an efficient mechanism for storing and searching
high-dimensional embeddings.

### 4. Retrieval-Augmented Generation

The project illustrates how retrieval can be inserted before LLM
generation.

### 5. Conversational AI

Chat history can be used to reinterpret follow-up questions.

### 6. Local LLM Inference

The project demonstrates that a relatively capable language model can be
run locally through CTransformers without requiring a commercial LLM
API.

### 7. Lightweight Deployment

Gradio provides a simple interface for turning a notebook experiment
into an interactive application.

------------------------------------------------------------------------

# ⚠️ Limitations

## 1. Narrow Knowledge Base

The chatbot is based on one Fibromyalgia information booklet.

It cannot reliably answer questions outside the scope of that source.

## 2. No Formal Evaluation

The notebook demonstrates example questions but does not report a
systematic evaluation using:

-   Retrieval precision
-   Recall
-   MRR
-   nDCG
-   Exact-match accuracy
-   ROUGE/BLEU
-   Faithfulness
-   Answer relevancy
-   Hallucination rate

A proper RAG evaluation framework would substantially strengthen the
project.

## 3. No Source Citations in the Answer

Although the conversational retrieval chain is configured with:

``` python
return_source_documents=True
```

the displayed answer is simply printed from:

``` python
result["answer"]
```

A production system should expose the retrieved source passages or
document/page references alongside the answer.

## 4. Medical Safety

The model can generate plausible but incorrect medical statements.

The system should therefore never be interpreted as a diagnostic or
treatment-decision engine.

## 5. Deployment Function Bypasses Retrieval

As discussed above, the final `rag_inference()` function calculates an
embedding but does not use it to query the FAISS database. It directly
invokes the LLaMA model with the raw query.
fileciteturn13file0L87-L112

This means the **final Gradio application is not fully
retrieval-grounded as currently implemented**.

## 6. Embedding Inconsistency

The FAISS database is constructed using:

``` text
BAAI/bge-base-en-v1.5
```

while the final Gradio inference uses:

``` text
all-MiniLM-L6-v2
```

for query encoding.

These should normally be kept consistent when performing vector
retrieval.

## 7. Large Model Footprint

The recorded LLaMA model is approximately:

``` text
3.79 GB
```

which makes local deployment considerably more resource-intensive than a
lightweight API-based application. fileciteturn10file0L67-L92

------------------------------------------------------------------------

# 🔧 Recommended Improvements

The project can be upgraded substantially while preserving its existing
architecture.

------------------------------------------------------------------------

## 1. Connect Gradio to the Actual Retriever

Instead of:

``` python
response = llm(query)
```

the deployment should call the previously constructed retrieval chain.

Conceptually:

``` text
Gradio
   │
   ▼
ConversationalRetrievalChain
   │
   ├── Query Reformulation
   ├── FAISS Retrieval
   └── LLaMA Generation
```

This would make the deployed application genuinely retrieval-augmented.

------------------------------------------------------------------------

## 2. Use One Embedding Model Consistently

Use the same embedding model for:

``` text
Document Embeddings
        +
Query Embeddings
```

For example:

``` text
BAAI/bge-base-en-v1.5
```

throughout the pipeline.

------------------------------------------------------------------------

## 3. Return Source Documents

The UI should display:

``` text
Answer
+
Retrieved Sources
+
Page Number
+
Document Title
```

This would significantly improve transparency and trust.

------------------------------------------------------------------------

## 4. Add a Medical Safety Prompt

The LLM should be explicitly instructed to:

-   Avoid diagnosis.
-   Avoid definitive treatment recommendations.
-   State uncertainty.
-   Encourage professional consultation where appropriate.
-   Escalate emergencies.
-   Stay within the supplied knowledge base.

------------------------------------------------------------------------

## 5. Add Retrieval Evaluation

Create a benchmark containing representative questions:

``` text
Question
Expected Source
Expected Answer
```

Then evaluate:

``` text
Retrieval Accuracy
Answer Relevancy
Faithfulness
Context Precision
Context Recall
```

------------------------------------------------------------------------

## 6. Add Hallucination Testing

Ask questions that are deliberately outside the source document.

The expected behavior should be:

``` text
Information not available in the supplied knowledge base.
```

rather than a fabricated answer.

------------------------------------------------------------------------

## 7. Improve Conversation Memory

The current notebook manually maintains:

``` python
HumanMessage
AIMessage
```

A production application should maintain session-specific conversation
histories and prevent histories from leaking between users.

------------------------------------------------------------------------

## 8. Modernize LangChain APIs

The notebook records deprecation warnings for some older LangChain
interfaces, including the `HuggingFaceEmbeddings` import used in the
project.

For a modern implementation, the appropriate newer LangChain
integrations should be used.

------------------------------------------------------------------------

# 🔐 Privacy and Data Considerations

The current prototype is comparatively privacy-friendly because the LLM
itself is run locally rather than through a hosted LLM API.

However, a production deployment still needs to consider:

-   User conversation storage
-   Server logs
-   Gradio deployment infrastructure
-   Uploaded documents
-   Personally identifiable information
-   Medical information
-   Access control
-   Encryption
-   Data retention

A medical chatbot should avoid storing sensitive conversations unless
there is a clear and appropriate privacy framework.

------------------------------------------------------------------------

# 📈 From Prototype to Production

A more robust architecture could look like:

``` text
                     User
                      │
                      ▼
               ┌─────────────┐
               │   Gradio /  │
               │     Web UI  │
               └──────┬──────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Safety / Policy │
             │    Layer        │
             └───────┬─────────┘
                     │
                     ▼
              Query Reformulation
                     │
                     ▼
              Embedding Model
                     │
                     ▼
              Vector Database
                     │
                     ▼
             Retrieved Context
                     │
                     ▼
             Context Validation
                     │
                     ▼
                Local LLM
                     │
                     ▼
            Grounded Answer
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      Source Pages        Safety Notice
```

This would make the application considerably more robust.

------------------------------------------------------------------------

# 🎓 Educational Value

This project is particularly valuable as a teaching and demonstration
project because it exposes almost every major component of a modern RAG
pipeline.

Students can observe the progression:

``` text
PDF
 ↓
Document Loader
 ↓
Text Chunks
 ↓
Embeddings
 ↓
Vector Database
 ↓
Retriever
 ↓
Prompt
 ↓
LLM
 ↓
Answer
 ↓
Gradio Application
```

It therefore provides a practical bridge between:

``` text
Natural Language Processing
          +
Information Retrieval
          +
Generative AI
          +
Large Language Models
          +
Application Development
```

------------------------------------------------------------------------

# 🔮 Future Extensions

Possible extensions include:

### Multi-document RAG

Add multiple trusted medical documents:

``` text
Fibromyalgia
Arthritis
Chronic Pain
Sleep
Exercise
Medication Information
```

### Hybrid Search

Combine:

``` text
Semantic Search
+
Keyword Search
```

to improve retrieval.

### Reranking

Use a cross-encoder reranker after FAISS retrieval.

### Source-Aware Answers

Return:

``` text
Answer
Source
Page
Relevant Passage
```

### Evaluation Dashboard

Track:

``` text
Retrieval Precision
Answer Faithfulness
Hallucination Rate
Latency
Token Usage
```

### Modern Local Models

Replace the older LLaMA 2 GGML model with a modern instruction-tuned
model supported by current inference frameworks.

### Persistent Vector Store

Save the FAISS index so that embeddings do not need to be recomputed
every time the application starts.

### API Layer

Separate the application into:

``` text
Frontend
    │
    ▼
Backend API
    │
    ├── Retriever
    ├── Safety Layer
    └── LLM
```

------------------------------------------------------------------------

# 📚 Source Document

The project's knowledge base is the supplied:

**Fibromyalgia --- Versus Arthritis**

The document provides patient-oriented information about the condition,
diagnosis, treatments, self-management, research, and additional
resources. Its contents explicitly include sections on symptoms,
diagnosis, treatments, physical and psychological therapies,
self-management, research, and further information.
fileciteturn18file0L32-L51

The document also makes clear that experiences can differ between
individuals and presents its information as a combination of
experiences, research, and facts. fileciteturn18file0L16-L30

------------------------------------------------------------------------

# 📁 Files in This Project

  -------------------------------------------------------------------------------------
  File                                              Role
  ------------------------------------------------- -----------------------------------
  `Medical_Chatbot_with_LLM.ipynb`                  Complete implementation notebook

  `fibromyalgia-information-booklet-july2021.pdf`   Knowledge-base document
  -------------------------------------------------------------------------------------

The notebook itself also downloads the large LLaMA model during
execution rather than requiring the model binary to be stored in the
repository. fileciteturn16file0L47-L84

------------------------------------------------------------------------

# ⭐ Key Takeaways

This project demonstrates a complete **domain-specific Generative AI
prototype** built around a medical knowledge source.

Its most important components are:

``` text
                  PDF Knowledge Base
                         │
                         ▼
                  Document Loading
                         │
                         ▼
                    Chunking
                         │
                         ▼
                  BGE Embeddings
                         │
                         ▼
                    FAISS
                         │
                         ▼
                    Retriever
                         │
                         ▼
              Conversational Retrieval
                         │
                         ▼
                    LLaMA 2
                         │
                         ▼
                     Answer
                         │
                         ▼
                     Gradio
```

The project is particularly significant because it demonstrates how an
LLM can be combined with an **external, domain-specific knowledge base**
rather than relying entirely on the model's internal parameters.

At the same time, the notebook provides a useful real-world lesson in
engineering: the **prototype RAG chain and the final Gradio deployment
are not completely equivalent**. The deployment function currently
bypasses the FAISS retriever, so the next logical step is to connect the
UI directly to the already-constructed conversational retrieval chain.

That distinction is important when evaluating the project---not only
because it identifies a limitation, but because it shows exactly how a
classroom/prototype RAG implementation can be evolved into a more
rigorous production-grade system.

------------------------------------------------------------------------

# 👤 Author

**Subhadeep Dey**

A Generative AI / NLP project exploring:

-   Large Language Models
-   Retrieval-Augmented Generation
-   Semantic Search
-   Vector Databases
-   Conversational AI
-   Local LLM Inference
-   Domain-Specific Question Answering

------------------------------------------------------------------------

## ⚠️ Disclaimer

This repository is an **educational and experimental AI project**.

It is **not a medical device, diagnostic system, or substitute for
professional medical advice**.

The chatbot may produce incomplete, inaccurate, outdated, or
inappropriate responses. Medical decisions should always be made with
qualified healthcare professionals and trusted, current clinical
resources.
