from fpdf import FPDF
import os

def create_ecas():
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Consolidated Account Statement (eCAS)", ln=True, align='C')
    pdf.ln(10)
    
    # Investor Details
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Investor Details:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt="Name: MANOJ SHARMA", ln=True)
    pdf.cell(200, 8, txt="PAN: ABCDE1234F", ln=True)
    pdf.cell(200, 8, txt="Email: manoj.sharma@example.com", ln=True)
    pdf.ln(10)
    
    # Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Portfolio Summary:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt="Total Value: INR 25,50,000", ln=True)
    pdf.cell(200, 8, txt="Equity: INR 18,00,000", ln=True)
    pdf.cell(200, 8, txt="Debt: INR 7,50,000", ln=True)
    pdf.ln(10)
    
    # Mutual Fund Holdings
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Mutual Fund Holdings:", ln=True)
    
    pdf.set_font("Arial", 'B', 10)
    # Header
    pdf.cell(80, 8, "Scheme Name", 1)
    pdf.cell(30, 8, "Folio No.", 1)
    pdf.cell(30, 8, "Units", 1)
    pdf.cell(40, 8, "Current Value", 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 10)
    funds = [
        ("HDFC Flexi Cap Fund - Direct", "123456789", "1200.50", "INR 8,50,000"),
        ("ICICI Prudential Bluechip", "987654321", "850.25", "INR 6,00,000"),
        ("SBI Small Cap Fund - Regular", "556677889", "500.00", "INR 3,50,000"),
        ("Kotak Liquid Fund - Growth", "223344556", "1500.67", "INR 7,50,000")
    ]
    
    for fund in funds:
        pdf.cell(80, 8, fund[0], 1)
        pdf.cell(30, 8, fund[1], 1)
        pdf.cell(30, 8, fund[2], 1)
        pdf.cell(40, 8, fund[3], 1)
        pdf.ln()
        
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.cell(200, 10, txt="* This is a computer-generated sample statement as of 02-Apr-2026.", ln=True, align='C')
    
    output_path = os.path.join(os.path.expanduser("~"), "Desktop", "sample_ecas.pdf")
    pdf.output(output_path)
    print(f"Sample eCAS generated successfully at: {output_path}")

if __name__ == "__main__":
    create_ecas()
