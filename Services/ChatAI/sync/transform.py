"""
Transform Zoho Contact to Document (V3 - COMPLETE Field Coverage)

Converts Zoho CRM contact data into structured documents for File Search.
Includes ALL 109 fields from the Zoho CRM Contacts module.
"""

from typing import Optional
from datetime import datetime


def transform_contact_to_document(contact: dict) -> dict:
    """
    Transform Zoho Contact into a structured document for RAG.
    
    This version captures ALL fields from Zoho CRM Contacts module.
    Complete field list: 109 fields across all custom modules/sections.
    """
    
    # Helper to safely get values
    def get(field, default=""):
        val = contact.get(field)
        if val is None or str(val).strip() in ("", "null", "None"):
            return default
        # Handle dict values (Owner, Created_By, etc.)
        if isinstance(val, dict):
            return val.get("name", str(val))
        return str(val).strip()
    
    # Helper for tags (list of dicts)
    def get_tags():
        tags_raw = contact.get("Tag")
        if tags_raw and isinstance(tags_raw, list):
            return ", ".join(t.get("name", "") for t in tags_raw if isinstance(t, dict))
        elif tags_raw:
            return str(tags_raw)
        return ""
    
    # =============================================
    # EXTRACT ALL FIELDS
    # =============================================
    
    # --- Basic Identity ---
    full_name = get("Full_Name") or f"{get('First_Name')} {get('Last_Name')}".strip()
    salutation = get("Salutation")
    first_name = get("First_Name")
    last_name = get("Last_Name")
    contact_name = get("Contact_Name") or full_name
    
    # --- Contact Details ---
    phone = get("Phone")
    mobile = get("Mobile")
    email = get("Email")
    secondary_email = get("Secondary_Email")
    email_opt_out = get("Email_Opt_Out")
    
    # --- Personal Details ---
    dob = get("Date_of_Birth") or get("Date_of_Birth_Actual") or get("Date_of_Birth_PAN")
    dob_actual = get("Date_of_Birth_Actual")
    age = get("Age")
    occupation = get("Occupation")
    qualification = get("Qualification")
    annual_income = get("Annual_Income")
    mothers_name = get("Mother_s_Name")
    place_of_birth = get("Place_of_Birth")
    alive = get("Alive")
    
    # --- Company/Organization ---
    company_name = get("Company_Name")
    company_type = get("Company_Type")
    account_name = get("Account_Name")
    department = get("Department")
    date_of_incorporation = get("Date_of_Incorporation")
    place_of_incorporation = get("Place_of_Incorporation")
    
    # --- Address (Primary) ---
    address_line = get("Address_Line") or get("Address_Line_1") or get("Mailing_Street")
    address_line2 = get("Address_Line2") or get("Address_Line_2")
    city = get("City") or get("City_Name") or get("Mailing_City")
    state = get("State") or get("State_UT") or get("Mailing_State")
    pin_code = get("Pin_Code") or get("ZIP_Postal_Code") or get("Mailing_Zip")
    country = get("Country") or get("Mailing_Country")
    communication_address = get("Communication_Address")
    location_home = get("Location_Home")
    location_office = get("Location_Office")
    
    # --- Address (Overseas) ---
    outside_country = get("Outside_Country")
    overseas_city = get("Overseas_City")
    overseas_state = get("Overseas_State")
    
    # --- Financial / KYC ---
    pan_no = get("PAN_NO") or get("PAN_Number")
    aadhaar = get("Aadhaar_No") or get("Aadhar_No")
    aum = get("AUM_Rs_Lakhs")
    tax_status = get("Tax_Status")
    currency = get("Currency")
    exchange_rate = get("Exchange_Rate")
    health_insurance = get("Health_Insurance")
    
    # --- HUF Details ---
    huf_pan = get("HUF_Pan_Number")
    annual_income_huf = get("Annual_Income_HUF")
    occupation_huf = get("Occupation_HUF")
    
    # --- Banking (Bank 1) ---
    bank_name = get("Bank_Name")
    account_no = get("Account_No") or get("Account_No_1")
    account_type = get("Account_Type")
    bank_type1 = get("Bank_Type1")
    ifsc = get("IFSC_Code")
    micr = get("MICR")
    
    # --- Banking (Bank 2) ---
    bank_name2 = get("Bank_Name_2")
    account_no2 = get("Account_No_2") or get("Account_No_21")
    bank_type2 = get("Bank_Type")
    ifsc2 = get("IFSC_Code_2")
    micr2 = get("MICR_2")
    upi_id = get("UPI_id")
    
    # --- CRM Classification ---
    contact_type = get("Contact_Type")
    category = get("Category_of_Client")
    lead_source = get("Lead_Source")
    lead_source_name = get("Lead_Source_Name")
    assigned_rm = get("Assigned_RM")
    owner = get("Owner")
    partner = get("Partner")
    reference_name = get("Reference_Name")
    referrer = get("Referrer")
    connected_to = get("Connected_To")
    associates = get("Associates")
    layout = get("Layout")
    
    # --- Tags & Segmentation ---
    tags = get_tags()
    segment_label = get("Segment_Label")
    segment_score = get("Segment_Score")
    
    # --- RFM Scores ---
    rfm_recency = get("Recency")
    rfm_frequency = get("Frequency")
    rfm_monetary = get("Monetary")
    
    # --- Portfolio & SIP ---
    portfolio_review_freq = get("Portfolio_Review_Frequency")
    portfolio_review_today = get("Portfolio_Review_Today")
    sip_topup = get("SIP_TopUp")
    sip_topup_date = get("SIP_TopUp_Date")
    sip_alarm_date = get("SIP_Alarm_date")
    
    # --- Engagement / Activity ---
    last_review = get("Last_Review")
    last_interaction = get("Last_Interaction_Date")
    last_activity_time = get("Last_Activity_Time")
    feedback_call = get("Feedback_Call")
    feedback_remarks = get("Feedback_Remarks")
    best_time = get("Best_Time_to_Call")
    who_can_call = get("Who_can_call")
    follow_ups = get("Follow_ups")
    follow_ups_count = get("Follow_ups_Count")
    
    # --- Messaging / Communication Stats ---
    messages_received = get("Messages_Received")
    messages_sent = get("Messages_Sent")
    messages_you_got = get("Messages_you_got")
    messages_you_sent = get("Messages_you_sent")
    total_messages = get("Total_Messages")
    last_message_send_by = get("Last_Message_send_by")
    time_since_last_msg = get("Time_Since_Last_Client_Message")
    avg_response_time = get("Average_Response_Time")
    first_response_time = get("First_Response_Time")
    inbound_clients_count = get("Inbound_Clients_Count")
    outbound_clients_count = get("Outbound_Clients_Count")
    
    # --- Subscription / Social ---
    social_lead_id = get("Social_Lead_ID")
    unsubscribed_mode = get("Unsubscribed_Mode")
    unsubscribed_time = get("Unsubscribed_Time")
    
    # --- Dates & Timestamps ---
    anniversary = get("Anniversary_Date")
    anniversary_today = get("Anniversary_Today")
    birthday_today = get("Birthday_Today")
    created_time = get("Created_Time")
    created_by = get("Created_By")
    modified_time = get("Modified_Time")
    modified_by = get("Modified_By")
    
    # --- Aggregate / Computed ---
    aggregate_field_1 = get("Aggregate_Field_1")
    
    # =============================================
    # BUILD STRUCTURED DOCUMENT
    # =============================================
    sections = []
    
    # Header
    sections.append(f"# Client Profile: {full_name}")
    
    # --- Contact Information ---
    contact_lines = ["\n## Contact Information"]
    contact_lines.append(f"- Full Name: {salutation + ' ' if salutation else ''}{full_name}")
    if phone: contact_lines.append(f"- Phone: {phone}")
    if mobile: contact_lines.append(f"- Mobile: {mobile}")
    if email: contact_lines.append(f"- Email: {email}")
    if secondary_email: contact_lines.append(f"- Secondary Email: {secondary_email}")
    if email_opt_out and email_opt_out.lower() not in ("false", "0"):
        contact_lines.append(f"- Email Opt Out: {email_opt_out}")
    sections.append("\n".join(contact_lines))
    
    # --- Personal Details ---
    personal_lines = ["\n## Personal Details"]
    has_personal = False
    if dob: personal_lines.append(f"- Date of Birth: {dob}"); has_personal = True
    if dob_actual and dob_actual != dob: personal_lines.append(f"- Date of Birth (Actual): {dob_actual}"); has_personal = True
    if age: personal_lines.append(f"- Age: {age}"); has_personal = True
    if occupation: personal_lines.append(f"- Occupation: {occupation}"); has_personal = True
    if qualification: personal_lines.append(f"- Qualification: {qualification}"); has_personal = True
    if annual_income: personal_lines.append(f"- Annual Income: {annual_income}"); has_personal = True
    if mothers_name: personal_lines.append(f"- Mother's Name: {mothers_name}"); has_personal = True
    if place_of_birth: personal_lines.append(f"- Place of Birth: {place_of_birth}"); has_personal = True
    if anniversary: personal_lines.append(f"- Anniversary Date: {anniversary}"); has_personal = True
    if anniversary_today: personal_lines.append(f"- Anniversary Today: {anniversary_today}"); has_personal = True
    if birthday_today: personal_lines.append(f"- Birthday Today: {birthday_today}"); has_personal = True
    if tax_status: personal_lines.append(f"- Tax Status: {tax_status}"); has_personal = True
    if alive: personal_lines.append(f"- Alive: {alive}"); has_personal = True
    if health_insurance: personal_lines.append(f"- Health Insurance: {health_insurance}"); has_personal = True
    if has_personal:
        sections.append("\n".join(personal_lines))
    
    # --- Company/Organization ---
    if company_name or account_name or department or place_of_incorporation:
        org_lines = ["\n## Organization"]
        if company_name: org_lines.append(f"- Company: {company_name}")
        if company_type: org_lines.append(f"- Company Type: {company_type}")
        if account_name: org_lines.append(f"- Account: {account_name}")
        if department: org_lines.append(f"- Department: {department}")
        if date_of_incorporation: org_lines.append(f"- Date of Incorporation: {date_of_incorporation}")
        if place_of_incorporation: org_lines.append(f"- Place of Incorporation: {place_of_incorporation}")
        sections.append("\n".join(org_lines))
    
    # --- Address ---
    addr_lines = ["\n## Address"]
    has_addr = False
    addr_parts = [p for p in [address_line, address_line2, city, state, pin_code, country] if p]
    if addr_parts:
        addr_lines.append(f"- Primary: {', '.join(addr_parts)}")
        has_addr = True
    if communication_address:
        addr_lines.append(f"- Communication Address: {communication_address}")
        has_addr = True
    if location_home:
        addr_lines.append(f"- Location (Home): {location_home}")
        has_addr = True
    if location_office:
        addr_lines.append(f"- Location (Office): {location_office}")
        has_addr = True
    # Overseas address
    overseas_parts = [p for p in [overseas_city, overseas_state, outside_country] if p]
    if overseas_parts:
        addr_lines.append(f"- Overseas Address: {', '.join(overseas_parts)}")
        has_addr = True
    if has_addr:
        sections.append("\n".join(addr_lines))
    
    # --- Financial / KYC Details ---
    fin_lines = ["\n## Financial & KYC Details"]
    has_fin = False
    if pan_no: fin_lines.append(f"- PAN Number: {pan_no}"); has_fin = True
    if aadhaar: fin_lines.append(f"- Aadhaar: {aadhaar}"); has_fin = True
    if aum: fin_lines.append(f"- AUM (Rs Lakhs): {aum}"); has_fin = True
    if currency and currency != "INR": fin_lines.append(f"- Currency: {currency}"); has_fin = True
    if exchange_rate and exchange_rate != "1": fin_lines.append(f"- Exchange Rate: {exchange_rate}"); has_fin = True
    if health_insurance: fin_lines.append(f"- Health Insurance: {health_insurance}"); has_fin = True
    if huf_pan: fin_lines.append(f"- HUF PAN: {huf_pan}"); has_fin = True
    if annual_income_huf: fin_lines.append(f"- Annual Income (HUF): {annual_income_huf}"); has_fin = True
    if occupation_huf: fin_lines.append(f"- Occupation (HUF): {occupation_huf}"); has_fin = True
    if has_fin:
        sections.append("\n".join(fin_lines))
    
    # --- Banking Details ---
    bank_lines = ["\n## Banking Details"]
    has_bank = False
    if bank_name:
        bank_lines.append(f"- Bank 1: {bank_name}")
        if bank_type1: bank_lines.append(f"  - Type: {bank_type1}")
        if account_no: bank_lines.append(f"  - Account No: {account_no}")
        if account_type: bank_lines.append(f"  - Account Type: {account_type}")
        if ifsc: bank_lines.append(f"  - IFSC: {ifsc}")
        if micr: bank_lines.append(f"  - MICR: {micr}")
        has_bank = True
    if bank_name2:
        bank_lines.append(f"- Bank 2: {bank_name2}")
        if bank_type2: bank_lines.append(f"  - Type: {bank_type2}")
        if account_no2: bank_lines.append(f"  - Account No: {account_no2}")
        if ifsc2: bank_lines.append(f"  - IFSC: {ifsc2}")
        if micr2: bank_lines.append(f"  - MICR: {micr2}")
        has_bank = True
    if upi_id: bank_lines.append(f"- UPI ID: {upi_id}"); has_bank = True
    if has_bank:
        sections.append("\n".join(bank_lines))
    
    # --- CRM Classification ---
    crm_lines = ["\n## CRM Classification"]
    has_crm = False
    if contact_type: crm_lines.append(f"- Contact Type: {contact_type}"); has_crm = True
    if category: crm_lines.append(f"- Category: {category}"); has_crm = True
    if lead_source: crm_lines.append(f"- Lead Source: {lead_source}"); has_crm = True
    if lead_source_name: crm_lines.append(f"- Lead Source Name: {lead_source_name}"); has_crm = True
    if assigned_rm: crm_lines.append(f"- Assigned RM: {assigned_rm}"); has_crm = True
    if owner: crm_lines.append(f"- Owner: {owner}"); has_crm = True
    if partner: crm_lines.append(f"- Partner: {partner}"); has_crm = True
    if reference_name: crm_lines.append(f"- Reference: {reference_name}"); has_crm = True
    if referrer: crm_lines.append(f"- Referrer: {referrer}"); has_crm = True
    if connected_to: crm_lines.append(f"- Connected To: {connected_to}"); has_crm = True
    if associates: crm_lines.append(f"- Associates: {associates}"); has_crm = True
    if tags: crm_lines.append(f"- Tags: {tags}"); has_crm = True
    if segment_label: crm_lines.append(f"- Segment: {segment_label}"); has_crm = True
    if segment_score: crm_lines.append(f"- Segment Score: {segment_score}"); has_crm = True
    if layout: crm_lines.append(f"- Layout: {layout}"); has_crm = True
    if social_lead_id: crm_lines.append(f"- Social Lead ID: {social_lead_id}"); has_crm = True
    if has_crm:
        sections.append("\n".join(crm_lines))
    
    # --- RFM Scoring ---
    rfm_lines = ["\n## RFM Scoring"]
    has_rfm = False
    if rfm_recency: rfm_lines.append(f"- Recency Score: {rfm_recency}"); has_rfm = True
    if rfm_frequency: rfm_lines.append(f"- Frequency Score: {rfm_frequency}"); has_rfm = True
    if rfm_monetary: rfm_lines.append(f"- Monetary Score: {rfm_monetary}"); has_rfm = True
    if has_rfm:
        sections.append("\n".join(rfm_lines))
    
    # --- Portfolio & SIP ---
    port_lines = ["\n## Portfolio & SIP"]
    has_port = False
    if portfolio_review_freq: port_lines.append(f"- Review Frequency: {portfolio_review_freq}"); has_port = True
    if portfolio_review_today: port_lines.append(f"- Portfolio Review Today: {portfolio_review_today}"); has_port = True
    if sip_topup: port_lines.append(f"- SIP TopUp %: {sip_topup}"); has_port = True
    if sip_topup_date: port_lines.append(f"- SIP TopUp Date: {sip_topup_date}"); has_port = True
    if sip_alarm_date: port_lines.append(f"- SIP Alarm Date: {sip_alarm_date}"); has_port = True
    if has_port:
        sections.append("\n".join(port_lines))
    
    # --- Engagement History ---
    eng_lines = ["\n## Engagement History"]
    has_eng = False
    if last_review: eng_lines.append(f"- Last Review: {last_review}"); has_eng = True
    if last_interaction: eng_lines.append(f"- Last Interaction: {last_interaction}"); has_eng = True
    if last_activity_time: eng_lines.append(f"- Last Activity Time: {last_activity_time}"); has_eng = True
    if feedback_call: eng_lines.append(f"- Feedback Call: {feedback_call}"); has_eng = True
    if feedback_remarks: eng_lines.append(f"- Feedback Remarks: {feedback_remarks}"); has_eng = True
    if best_time: eng_lines.append(f"- Best Time to Call: {best_time}"); has_eng = True
    if who_can_call: eng_lines.append(f"- Who Can Call: {who_can_call}"); has_eng = True
    if follow_ups: eng_lines.append(f"- Follow Ups: {follow_ups}"); has_eng = True
    if follow_ups_count: eng_lines.append(f"- Follow-ups Count: {follow_ups_count}"); has_eng = True
    if has_eng:
        sections.append("\n".join(eng_lines))
    
    # --- Messaging & Communication Stats ---
    msg_lines = ["\n## Messaging & Communication"]
    has_msg = False
    if messages_sent: msg_lines.append(f"- Messages Sent: {messages_sent}"); has_msg = True
    if messages_received: msg_lines.append(f"- Messages Received: {messages_received}"); has_msg = True
    if messages_you_sent: msg_lines.append(f"- Messages You Sent: {messages_you_sent}"); has_msg = True
    if messages_you_got: msg_lines.append(f"- Messages You Got: {messages_you_got}"); has_msg = True
    if total_messages: msg_lines.append(f"- Total Messages: {total_messages}"); has_msg = True
    if last_message_send_by: msg_lines.append(f"- Last Message Send By: {last_message_send_by}"); has_msg = True
    if time_since_last_msg: msg_lines.append(f"- Time Since Last Client Message: {time_since_last_msg}"); has_msg = True
    if avg_response_time: msg_lines.append(f"- Average Response Time: {avg_response_time}"); has_msg = True
    if first_response_time: msg_lines.append(f"- First Response Time: {first_response_time}"); has_msg = True
    if inbound_clients_count: msg_lines.append(f"- Inbound Clients Count: {inbound_clients_count}"); has_msg = True
    if outbound_clients_count: msg_lines.append(f"- Outbound Clients Count: {outbound_clients_count}"); has_msg = True
    if has_msg:
        sections.append("\n".join(msg_lines))
    
    # --- Subscription Status ---
    sub_lines = ["\n## Subscription"]
    has_sub = False
    if unsubscribed_mode: sub_lines.append(f"- Unsubscribed Mode: {unsubscribed_mode}"); has_sub = True
    if unsubscribed_time: sub_lines.append(f"- Unsubscribed Time: {unsubscribed_time}"); has_sub = True
    if has_sub:
        sections.append("\n".join(sub_lines))
    
    # --- Aggregates / Computed ---
    if aggregate_field_1:
        sections.append(f"\n## Aggregates\n- Aggregate Field 1: {aggregate_field_1}")
    
    # --- Footer / Record Info ---
    footer_lines = ["\n---"]
    if created_by: footer_lines.append(f"Created By: {created_by}")
    footer_lines.append(f"Record Created: {created_time}")
    if modified_by: footer_lines.append(f"Modified By: {modified_by}")
    footer_lines.append(f"Last Updated: {modified_time}")
    sections.append("\n".join(footer_lines))
    
    content = "\n".join(sections)
    
    return {
        "id": contact.get("id", ""),
        "content": content,
        "metadata": {
            "client_id": contact.get("id"),
            "client_name": full_name,
            "email": email,
            "phone": phone or mobile,
            "city": city,
            "category": category,
            "source": "zoho_crm",
            "last_updated": modified_time or datetime.now().isoformat(),
        }
    }
