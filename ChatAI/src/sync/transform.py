"""
Transform Zoho Contact to Document

Converts Zoho CRM contact data into structured documents for File Search.
"""

from typing import Optional
from datetime import datetime


def transform_contact_to_document(contact: dict) -> dict:
    """
    Transform Zoho Contact into a structured document for RAG.
    
    Args:
        contact: Zoho contact data dictionary
    
    Returns:
        Document dict with content and metadata
        
    Expected Zoho fields (customize based on your Zoho setup):
    - Full_Name, First_Name, Last_Name
    - Phone, Mobile, Email
    - Portfolio_Holdings (custom field)
    - Investment_Goals (custom field)
    - Risk_Profile (custom field)
    - Total_Investment (custom field)
    - SIP_Amount (custom field)
    """
    
    # Extract basic info
    full_name = contact.get("Full_Name") or f"{contact.get('First_Name', '')} {contact.get('Last_Name', '')}".strip()
    phone = contact.get("Phone") or contact.get("Mobile", "")
    email = contact.get("Email", "")
    
    # Extract custom financial fields
    portfolio = contact.get("Portfolio_Holdings", "Not recorded")
    goals = contact.get("Investment_Goals", "Not specified")
    risk_profile = contact.get("Risk_Profile", "Medium")
    total_investment = contact.get("Total_Investment", "Not available")
    sip_amount = contact.get("SIP_Amount", "Not available")
    
    # Format transactions if available
    transactions = format_transactions(contact.get("Transactions", []))
    
    # Build structured content for RAG
    content = f"""# Client Profile: {full_name}

## Contact Information
- Name: {full_name}
- Phone: {phone}
- Email: {email}

## Financial Profile
- Risk Profile: {risk_profile}
- Total Investment: {total_investment}
- Monthly SIP: {sip_amount}

## Investment Goals
{goals}

## Portfolio Holdings
{portfolio}

## Recent Transactions
{transactions}

---
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    
    return {
        "id": contact.get("id", ""),
        "content": content,
        "metadata": {
            "client_id": contact.get("id"),
            "client_name": full_name,
            "risk_profile": risk_profile,
            "last_updated": contact.get("Modified_Time", datetime.now().isoformat()),
            "source": "zoho_crm"
        }
    }


def format_transactions(transactions: list) -> str:
    """
    Format transaction list for document.
    
    Args:
        transactions: List of transaction dicts
    
    Returns:
        Formatted string of recent transactions
    """
    if not transactions:
        return "No recent transactions recorded"
    
    lines = []
    for txn in transactions[:10]:  # Limit to 10 most recent
        date = txn.get("date", "N/A")
        txn_type = txn.get("type", "N/A")  # SIP, Lumpsum, Redemption
        scheme = txn.get("scheme_name", "N/A")
        amount = txn.get("amount", "N/A")
        
        lines.append(f"- {date}: {txn_type} - {scheme} - ₹{amount}")
    
    return "\n".join(lines) if lines else "No recent transactions recorded"


def format_holdings(holdings: list) -> str:
    """
    Format portfolio holdings for document.
    
    Args:
        holdings: List of holding dicts
    
    Returns:
        Formatted string of holdings
    """
    if not holdings:
        return "No holdings recorded"
    
    lines = ["| Scheme | Category | Value | % of Portfolio |"]
    lines.append("|--------|----------|-------|----------------|")
    
    for h in holdings:
        scheme = h.get("scheme_name", "N/A")
        category = h.get("category", "N/A")
        value = h.get("current_value", "N/A")
        percentage = h.get("allocation_pct", "N/A")
        lines.append(f"| {scheme} | {category} | ₹{value} | {percentage}% |")
    
    return "\n".join(lines)
