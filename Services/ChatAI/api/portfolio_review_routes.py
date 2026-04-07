from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import PyPDF2
from google import genai
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/api/portfolio/analyze")
async def analyze_portfolio(
    file: UploadFile = File(...),
    password: str = Form(None)
):
    try:
        # Read file
        content = await file.read()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name
            
        unencrypted_pdf_path = temp_pdf_path
        
        # If password is provided, try to decrypt
        if password:
            unencrypted_pdf_path = tempfile.mktemp(suffix=".pdf")
            with open(temp_pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if reader.is_encrypted:
                    try:
                        reader.decrypt(password)
                    except Exception as e:
                        raise HTTPException(status_code=400, detail="Invalid password for eCAS PDF.")
                    
                    writer = PyPDF2.PdfWriter()
                    for page in reader.pages:
                        writer.add_page(page)
                        
                    with open(unencrypted_pdf_path, 'wb') as f_out:
                        writer.write(f_out)
                else:
                    unencrypted_pdf_path = temp_pdf_path
        else:
            with open(temp_pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if reader.is_encrypted:
                    raise HTTPException(status_code=400, detail="PDF is encrypted but no password provided.")
        
        # Upload to Gemini
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        uploaded_file = gemini_client.files.upload(file=unencrypted_pdf_path)
        
        prompt = """
        You are an expert wealth manager and portfolio analyst.
        Attached is an eCAS (Consolidated Account Statement) for a client's mutual fund and equity portfolio.
        Analyze the client's portfolio and provide a comprehensive Portfolio Review.
        Include:
        1. **Asset Allocation Summary**: Breakdown of equity, debt, liquid, etc.
        2. **Top Holdings**: Noticeable or large holdings.
        3. **Risk Profile Assessment**: What does their portfolio indicate about their risk appetite.
        4. **Insights & Recommendations**: Strengths of the portfolio and areas for diversification or rebalancing.
        Format beautifully in Markdown with tables where appropriate.
        """
        
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[uploaded_file, prompt]
        )
        
        # Cleanup temp files
        try:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            if unencrypted_pdf_path != temp_pdf_path and os.path.exists(unencrypted_pdf_path):
                os.remove(unencrypted_pdf_path)
            gemini_client.files.delete(name=uploaded_file.name)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")
            
        return JSONResponse({"analysis": response.text})
        
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        logger.error(f"Error analyzing portfolio: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
