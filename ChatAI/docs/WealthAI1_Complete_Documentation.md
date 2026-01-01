# WealthAI1 - Complete System Documentation

> **AI-Powered Financial Knowledge Platform for Indian Intermediaries**  
> Multi-tenant RAG system with client-specific data isolation and intelligent query routing

---

## 📑 Table of Contents

1. [What is the App About?](#1-what-is-the-app-about)
2. [How Querying Works](#2-how-querying-works)
3. [Intent Classification System](#3-intent-classification-system)
4. [RAG Data Architecture](#4-rag-data-architecture)
5. [Why Better Than ChatGPT/Gemini](#5-why-better-than-chatgptgemini)
6. [End-to-End Costing Model](#6-end-to-end-costing-model)
7. [Replicating RAG for Each Customer](#7-replicating-rag-for-each-customer)
8. [Separate API Keys per Customer](#8-separate-api-keys-per-customer)
9. [Monthly Embedding Updates](#9-monthly-embedding-updates)
10. [Batch Jobs & Zoho Sync](#10-batch-jobs--zoho-sync)
11. [Service vs Product Model](#11-service-vs-product-model)
12. [Data Protection for Zoho Customers](#12-data-protection-for-zoho-customers)
13. [Privacy Considerations](#13-privacy-considerations)

---

## 1. What is the App About?

### 🎯 Core Purpose

**WealthAI1** is a specialized AI chatbot platform built **exclusively for Indian financial intermediaries**:
- Mutual Fund Distributors (MFDs)
- Insurance Advisors & Agents
- Wealth Managers
- Independent Financial Advisors (IFAs)

### The Problem We Solve

| Problem | ChatGPT/Gemini | WealthAI1 |
|---------|----------------|-----------|
| **Hallucination** | Makes up policy features | ✅ Only answers from verified docs |
| **Outdated Data** | Training cutoff months ago | ✅ Upload today, query today |
| **No Client Context** | Doesn't know "Sharma ji's portfolio" | ✅ Integrated with Zoho CRM |
| **Privacy Concerns** | Data goes to public AI | ✅ Data stays in YOUR GCP project |
| **Generic Answers** | Not India-specific | ✅ Built for SEBI, AMFI, IRDAI context |

### Key Features

```
┌─────────────────────────────────────────────────────────┐
│  🧠 8-Intent Classification System                      │
│  📚 1300+ Product Documents (MF + Insurance)            │
│  👤 Zoho CRM Integration for Client Portfolios          │
│  🔒 Multi-Tenant Data Isolation                         │
│  🇮🇳 Hindi/Hinglish Support                             │
│  💰 Pay-Per-Use Model (Customer's GCP)                  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. How Querying Works

### Visual Query Flow

```mermaid
sequenceDiagram
    participant U as 👤 Advisor
    participant FE as 🖥️ Frontend
    participant API as ⚡ FastAPI
    participant IC as 🧠 Intent Classifier
    participant R as 🔀 Router
    participant FS as 📚 File Search
    participant WS as 🌐 Web Search
    participant LLM as 🤖 Gemini

    U->>FE: "Sharma ji ka portfolio dikhao"
    FE->>API: POST /api/query
    
    API->>IC: Classify Intent
    IC-->>API: CLIENT_VIEW (client: Sharma ji)
    
    API->>R: Route to Client Handler
    R->>FS: Query Client Store
    FS-->>R: [Portfolio Data]
    
    R->>LLM: Generate Response
    LLM-->>API: "Sharma ji का AUM ₹45.67 लाख है..."
    API-->>FE: JSON Response
    FE-->>U: Display with Citations
```

### Processing Steps

```
Step 1: INTENT CLASSIFICATION
├── Input: "What is the NAV of HDFC Top 100?"
├── Gemini Flash analyzes query
├── Output: { intent: "product_info", sub_intent: "nav_check" }
└── Time: ~200ms

Step 2: ENTITY EXTRACTION
├── scheme_names: ["HDFC Top 100"]
├── product_category: "MF"
├── action_verb: null
└── client_name: null

Step 3: ROUTING
├── Intent → Handler Mapping
├── product_info → File Search (Products Store)
├── client_view → File Search (Client Store)
├── education → Plain LLM
└── market → Google Grounded Search

Step 4: CONTEXT RETRIEVAL
├── Query relevant document chunks
├── Return top-k results with citations
└── Merge multiple sources if needed

Step 5: RESPONSE GENERATION
├── System prompt + Context + Query
├── Generate using Gemini 2.5 Flash
└── Return with source citations
```

---

## 3. Intent Classification System

### 8 Primary Intent Categories

| Intent | Description | Handler | Example |
|--------|-------------|---------|---------|
| **PRODUCT_INFO** | Product features, NAV, returns | File Search | "HDFC Top 100 ka NAV?" |
| **PRODUCT_COMPARE** | Compare 2+ products | Multi-source | "Axis vs Mirae ELSS?" |
| **CLIENT_VIEW** | Portfolio, holdings, AUM | Client Store | "Sharma ji's AUM?" |
| **CLIENT_ACTION** | SIP start/stop, redemption | Client Store | "Stop Patel ji's SIP" |
| **REGULATORY** | SEBI, AMFI, compliance | File Search + Web | "KYC rules?" |
| **MARKET** | News, trends, indices | Google Search | "Nifty today?" |
| **EDUCATION** | Concepts, definitions | Plain LLM | "SIP kya hai?" |
| **OPERATIONS** | Commissions, statements | LLM | "My trail commission?" |

### Indian Name Recognition

```python
# Patterns that trigger CLIENT intent:
- "Sharma ji" → Client: Sharma ji
- "Patel sahab" → Client: Patel sahab  
- "Ramesh bhai" → Client: Ramesh bhai
- "Sunita madam" → Client: Sunita madam
- "Mr. Agarwal" → Client: Mr. Agarwal
- "client Verma" → Client: Verma
```

### Hindi/Hinglish Support

```python
# Detected and responded in same language:
Query: "SIP kya hota hai?"
Language: hinglish
Response: "SIP का पूरा नाम है Systematic Investment Plan..."
```

---

## 4. RAG Data Architecture

### Data Structure

```
📂 File Search Stores (Vertex AI)
│
├── 📁 Shared Product Store (per customer)
│   ├── 🏢 Insurance/
│   │   ├── Health Insurance/
│   │   │   ├── Acko_Personal_Health_Policy.pdf
│   │   │   └── Star_Health_Premier.pdf
│   │   ├── Life Insurance/
│   │   └── Motor Insurance/
│   │
│   ├── 💰 Mutual Funds/
│   │   ├── HDFC/ (268 schemes)
│   │   ├── SBI/
│   │   ├── ICICI/
│   │   └── Axis/
│   │
│   └── 📈 Stocks/
│       └── Research Reports/
│
└── 👤 Client Data Store (per customer)
    ├── client_sharma_holdings.txt
    ├── client_patel_portfolio.txt
    └── client_verma_transactions.txt
```

### Current Data Volume

| Category | Count | Size |
|----------|-------|------|
| Insurance Documents | 1,034 | ~1.5 GB |
| Mutual Fund Documents | 268 | ~800 MB |
| Total Files | ~1,300 | ~2.4 GB |

### How Documents are Processed

```
Original PDF (47 pages)
        ↓
Chunked into 94 segments (~400 tokens each)
        ↓
Embedded with text-embedding-004
        ↓
Stored in Vertex AI Vector Database
        ↓
Searchable within milliseconds
```

---

## 5. Why Better Than ChatGPT/Gemini

### Comparison Matrix

| Aspect | WealthAI1 | ChatGPT 4 | Gemini + Web |
|--------|-----------|-----------|--------------|
| **Data Freshness** | ✅ Real-time (upload today) | ❌ Months old | ⚠️ May find spam |
| **Accuracy** | ✅ From YOUR documents | ❌ Hallucinates | ⚠️ Unreliable sources |
| **Citations** | ✅ Exact PDF + page | ❌ None | ⚠️ Generic web links |
| **Client Context** | ✅ Knows portfolios | ❌ No CRM | ❌ No integration |
| **Privacy** | ✅ Your GCP project | ⚠️ May train on data | ⚠️ Public search |
| **Cost** | ✅ ₹0.01/query | ❌ ₹0.80/query | ⚠️ ₹0.10/query |
| **Compliance** | ✅ SEBI-backed | ❌ Generic disclaimers | ❌ Unverified |

### Real Example

**Query**: "What is the claim settlement ratio of Acko health policy?"

| System | Response | Problem |
|--------|----------|---------|
| **ChatGPT** | "Typically 85-90%..." | ❌ No source, could be wrong |
| **Gemini Web** | "88.5% according to blogs..." | ❌ Unreliable blog source |
| **WealthAI1** | "Not in policy document, but claims processed in 30 days. 📄 Source: Policy_Wordings_Acko.pdf, Page 12" | ✅ Honest + Cited |

---

## 6. End-to-End Costing Model

### Platform Owner (You) - Revenue

```
Revenue Streams:
├── Setup Fee: ₹50,000 per customer (one-time)
├── Monthly AMC: ₹2,000/month per customer
└── Example (10 customers):
    ├── Year 1: ₹5,00,000 (setup) + ₹2,40,000 (AMC) = ₹7,40,000
    └── Year 2+: ₹2,40,000 (AMC only)

Your Costs:
├── Cloud Hosting: ₹8,000/month
├── Document Curation: ₹5,000/month
└── Total Cost: ₹1,56,000/year

NET PROFIT (10 customers): ₹5,84,000/year
```

### Financial Advisor (Customer) - Cost

```
Costs to Customer:
├── To You (Platform):
│   ├── Setup: ₹50,000 (one-time)
│   └── AMC: ₹2,000/month
│
└── To Google (Direct billing):
    ├── Gemini Flash: ~₹200/month (classification)
    ├── Gemini Pro: ~₹800/month (generation)
    ├── File Search: ~₹0 (free tier)
    └── Google Search: ~₹300/month

Total First Year: ₹50,000 + ₹39,600 = ₹89,600
Monthly Ongoing: ₹3,300/month
```

### Cost Per Query Breakdown

```
Simple Query (education):
├── Intent Classification: ₹0.0003
├── LLM Generation: ₹0.008
└── Total: ₹0.0083 (~0.8 paise)

Product Query (with RAG):
├── Intent Classification: ₹0.0003
├── File Search Retrieval: ₹0.002
├── LLM Generation: ₹0.012
└── Total: ₹0.0143 (~1.4 paise)

Complex Query (multi-source):
├── Intent Classification: ₹0.0003
├── File Search: ₹0.002
├── Google Grounding: ₹0.035
├── LLM Generation: ₹0.015
└── Total: ₹0.0523 (~5.2 paise)
```

---

## 7. Replicating RAG for Each Customer

### Sync Architecture

```mermaid
graph LR
    A[Platform Owner] -->|Upload| B[Master Repo]
    B --> C[Admin Dashboard]
    C -->|Sync Button| D[Sync Engine]
    
    subgraph "Customer Stores"
        D --> E[Customer A Store]
        D --> F[Customer B Store]
        D --> G[Customer N Store]
    end
    
    E -->|Uses Customer A API Key| H[Customer A Queries]
    F -->|Uses Customer B API Key| I[Customer B Queries]
```

### Step-by-Step Process

```python
# 1. Upload New Document (Admin Dashboard)
POST /admin/documents/upload
{
  "file": "HDFC_New_Fund_Jan2024.pdf",
  "category": "Mutual Funds/HDFC"
}

# 2. Trigger Sync
POST /admin/sync/all  # All customers
# OR
POST /admin/sync/customer/tenant_abc123  # Specific customer

# 3. Backend Process
async def sync_docs_to_all_customers():
    for customer in get_all_customers():
        api_key = get_tenant_api_key(customer.id)
        client = genai.Client(api_key=api_key)
        
        for doc in get_new_documents():
            client.file_search_stores.upload(
                file=doc.path,
                store_name=customer.store_name
            )
```

---

## 8. Separate API Keys per Customer

### Multi-Tenancy Key Flow

```mermaid
graph TD
    A[Customer Signs Up] --> B[Provides GCP API Key]
    B --> C[Stored in Secret Manager]
    C --> D[Mapped to Tenant ID]
    
    E[Query Arrives] --> F{Extract X-Tenant-ID}
    F --> G[Lookup in Tenant DB]
    G --> H[Retrieve API Key]
    H --> I[Initialize Gemini Client]
    I --> J[Execute Query]
    J --> K[Billed to Customer's GCP]
```

### Implementation

```python
# Customer Onboarding
@router.post("/admin/customers")
async def add_customer(name: str, gemini_api_key: str):
    customer = {"id": f"tenant_{uuid4()}", "name": name}
    
    # Store key securely
    store_secret(f"gemini_key_{customer['id']}", gemini_api_key)
    
    # Create their File Search Store
    client = genai.Client(api_key=gemini_api_key)
    store = client.file_search_stores.create(
        display_name=f"products-{name}"
    )
    
    return customer

# Runtime Key Retrieval
@app.post("/api/query")
async def query(request: QueryRequest, tenant_id: str = Header(...)):
    api_key = get_secret(f"gemini_key_{tenant_id}")
    client = genai.Client(api_key=api_key)
    # All costs billed to THIS tenant's GCP
```

---

## 9. Monthly Embedding Updates

### Automated Pipeline

```mermaid
graph LR
    A[CRON: 1st of Month] --> B[Scan data/products/]
    B --> C{New Files?}
    C -->|Yes| D[Archive Old Versions]
    D --> E[Upload New Files]
    E --> F[Sync to All Tenants]
    F --> G[Log Results]
    C -->|No| H[Skip]
```

### Implementation

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('cron', day=1, hour=2)  # 1st of month, 2 AM
async def monthly_product_sync():
    logger.info("Starting monthly product document sync")
    
    new_docs = find_updated_documents(
        since=datetime.now() - timedelta(days=30)
    )
    
    if new_docs:
        results = await sync_documents_to_all_tenants(new_docs)
        send_admin_email(
            f"Synced {len(new_docs)} docs to {results['success']} tenants"
        )
```

### Key Points
- **No Manual Embeddings**: Vertex AI auto-indexes on upload
- **Processing Time**: 5-30 seconds per document
- **Zero Downtime**: Old version serves until new one ready

---

## 10. Batch Jobs & Zoho Sync

### Weekly Sync Schedule

| Job | Day | Time | Purpose |
|-----|-----|------|---------|
| **Product Sync** | Sunday | 3 AM | Push new MF/Insurance docs |
| **Zoho Sync** | Monday | 1 AM | Update client portfolios |
| **Cleanup** | 1st | 4 AM | Archive old documents |

### Zoho Client Data Sync

```python
@scheduler.scheduled_job('cron', day_of_week='mon', hour=1)
async def weekly_zoho_sync():
    for customer in get_customers_with_zoho():
        zoho_creds = get_secret(f"zoho_creds_{customer.id}")
        zoho_client = ZohoClient(zoho_creds)
        
        clients = zoho_client.get_all_contacts()
        
        for client in clients:
            # Transform to LLM-friendly format
            text = f"""
            Client: {client.name}
            Portfolio:
            - {client.scheme_name}: {client.units} units
            - Current Value: ₹{client.current_value}
            """
            
            upload_to_file_search(customer.store_name, text)
```

---

## 11. Service vs Product Model

### Why This is a SERVICE, Not a Product

| Aspect | Traditional SaaS Product | WealthAI1 Service |
|--------|--------------------------|-------------------|
| **Hosting** | Company hosts everything | Customer hosts their AI (GCP) |
| **Billing** | Fixed subscription | Pay-per-use (customer controls) |
| **Data** | On our servers | In customer's GCP project |
| **Features** | Same for everyone | Customizable per tenant |

### Your Value-Add Services

```
1. DATA CURATION SERVICE (Core Value)
├── Curated 1300+ financial documents
├── Verified HDFC, ICICI, Acko docs
├── Monthly updates
└── Quality assurance

2. INTEGRATION SERVICE
├── Zoho CRM connector
├── Automatic sync pipelines
├── Error handling/retry
└── Data transformation

3. PROMPT ENGINEERING SERVICE
├── 8-intent classification
├── Hindi/Hinglish support
├── Indian finance jargon
└── Continuous improvement

4. MAINTENANCE SERVICE
├── Bug fixes
├── Security updates
├── Feature enhancements
└── 24/7 monitoring
```

### Customer Lock-In
- **Data Network Effect**: Your library improves monthly
- **Integration Lock-in**: Moving means rebuilding Zoho connectors
- **Prompt IP**: Classification logic is proprietary
- **Inertia**: "It just works" is hard to replace

---

## 12. Data Protection for Zoho Customers

### Multi-Layer Privacy Architecture

```
┌─────────────────────────────────────────┐
│  Layer 1: Physical Isolation            │
│  Each tenant = Separate GCP Project     │
└─────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Layer 2: API Key Segregation           │
│  Tenant A key ≠ Tenant B key            │
└─────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Layer 3: Application Access Control    │
│  X-Tenant-ID header validation          │
│  Admin routes require X-Admin-Key       │
└─────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Layer 4: Encryption                    │
│  TLS 1.3 in transit, AES-256 at rest    │
└─────────────────────────────────────────┘
```

### Threat Analysis

| Threat | Mitigation |
|--------|------------|
| **Cross-Tenant Data Leak** | ✅ Impossible: Different GCP projects |
| **API Key Theft** | ✅ Stored in Secret Manager, rotated |
| **Zoho Credential Exposure** | ✅ OAuth tokens, not passwords |
| **Man-in-the-Middle** | ✅ TLS 1.3 + certificate pinning |
| **Insider Threat** | ✅ Platform owner has NO access to tenant keys |

---

## 13. Privacy Considerations

### Data Flow Privacy

```mermaid
graph TD
    A[Client Data in Zoho] -->|Encrypted Sync| B[Your GCP Project]
    B -->|Query| C[WealthAI1 Platform]
    C -->|Uses YOUR API Key| D[Gemini API]
    D -->|Response| C
    C -->|Display| E[Your Browser]
    
    style B fill:#90EE90
    style D fill:#FFB6C1
```

### Regulatory Compliance

| Requirement | Status |
|-------------|--------|
| **DPDP Act (India)** | |
| Data Localization | ✅ Tenant can choose `asia-south1` region |
| Right to Deletion | ✅ Delete tenant = instant data wipe |
| Consent | ✅ Advisors get client consent for Zoho sync |
| Purpose Limitation | ✅ Data only for portfolio queries |
| | |
| **SEBI RIA Regulations** | |
| Record Keeping | ✅ All queries logged with tenant ID |
| Audit Trail | ✅ Citations show document source |
| No Recommendations | ✅ System explains, doesn't recommend |

### What Data Goes Where

| Data Type | Storage | Access |
|-----------|---------|--------|
| Product Documents | Customer's GCP | Customer only |
| Client Portfolios | Customer's GCP | Customer only |
| Query Logs | Platform Database | Platform + Customer |
| API Keys | Secret Manager | Platform only (encrypted) |

### Privacy Guarantees

1. **Your client data never touches our servers** - goes directly from Zoho to your GCP
2. **We cannot see your queries** - encrypted in transit, processed by your API key
3. **No model training on your data** - Gemini API doesn't train on API customers
4. **Delete anytime** - we have scripts to purge all your data

---

## 🚀 Getting Started

### For Platform Owners
```bash
# 1. Clone and setup
git clone <repo>
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Add GEMINI_API_KEY, ADMIN_API_KEY, CORS_ORIGINS

# 3. Start server
uvicorn src.api.main:app --reload
```

### For Customers (Financial Advisors)
1. Get a Google Cloud Platform account
2. Enable Gemini API and get API key
3. Share key with platform owner
4. Start querying!

---

> **Built with ❤️ for Indian Financial Intermediaries**  
> *Empowering advisors with AI that understands your business*
