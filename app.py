import os
from flask import Flask, request, jsonify, render_template, session
import google.generativeai as genai
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-for-sessions'  # Change this in production

# API Keys (hardcoded as requested)
OPENAI_API_KEY = "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"
GROK_API_KEY = "xai-3xayae3jzDIYripTNLPXhPs7p8ZEgj84RJb8pZS6jYQ00YXQLBInLEOMvYG1FX4GUfrgj0vS2Y9VWQqD"
# GEMINI_API_KEY = "AIzaSyAqa_CfOjzRtjzW7ncqki1cnLZhV1Ky8-4"
GEMINI_API_KEY = "AIzaSyBfRb8dRVthgwQV8-_bbCVjYx1J9v2goNI"

# Business analysis prompts for Vamo
BUSINESS_ANALYSIS_PROMPTS = {
    "VAMO_V1": """You are an expert at analyzing companies and startups. Provide an analysis of the company based on their website content. Include a YC-style punchy one-liner that captures what the company does, and a provocative question that highlights the core problem they're solving - think if someone is pitching the company on shark tank, what would capture the attention of the sharks? 

Examples of provocative questions:
- 'Why are we teaching students how to dissect frogs… but not how to do their taxes?'
- 'Sick of cookie-cutter hotel rooms with personality? So were 150 million people.'
- 'What if you never had to download music again?'
- 'How powerful could you be if you stopped numbing yourself by fapping every time you felt bored, lonely, or anxious?'

Extract founder details if available on the website (names, titles, avatar URLs). Also determine if the business is actively operating and the content matches the company name/tagline - watch out for domain parking pages, unrelated redirects, or placeholder content."""
}

# Schema for structured business analysis output - matches actual Vamo schema
BUSINESS_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "details": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the company (as it appears on the website)"},
                "description": {"type": "string", "description": "Brief description of the company (two sentences). Make sure this is PUNCHY. YC style one-liners. Not boring."},
                "catchy_one_liner": {"type": "string", "description": "YC-style punchy one-liner that captures the essence of what the company does"},
                "provocative_question": {"type": "string", "description": "Provocative question that highlights the problem the business is trying to solve - think if someone is pitching the company on shark tank, what would capture the attention of the sharks? Examples: 'Why are we teaching students how to dissect frogs… but not how to do their taxes?' 'Sick of cookie-cutter hotel rooms with personality? So were 150 million people.', 'What if you never had to download music again?', 'How powerful could you be if you stopped numbing yourself by fapping every time you felt bored, lonely, or anxious?'"}
            },
            "required": ["name", "description", "catchy_one_liner", "provocative_question"],
            "additionalProperties": False
        },
        "valuation": {
            "type": "object",
            "properties": {
                "estimate": {"type": "string", "description": "Estimated valuation range of the company"},
                "confidence": {"type": "number", "description": "Confidence in the valuation (0.0-1.0)"}
            },
            "required": ["estimate", "confidence"],
            "additionalProperties": False
        },
        "pros": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title of the pro"},
                    "description": {"type": "string", "description": "Detailed description of the pro"}
                },
                "required": ["title", "description"],
                "additionalProperties": False
            },
            "description": "List of positive aspects about the company"
        },
        "concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title of the concern"},
                    "description": {"type": "string", "description": "Detailed description of the concern"},
                    "severity": {"type": "integer", "description": "Severity of the concern (1-10)"}
                },
                "required": ["title", "description", "severity"],
                "additionalProperties": False
            },
            "description": "List of potential concerns or risks"
        },
        "competitors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the competitor"},
                    "description": {"type": "string", "description": "Brief description of the competitor"},
                    "threat_level": {"type": "integer", "description": "Threat level posed by this competitor (1-10)"}
                },
                "required": ["name", "description", "threat_level"],
                "additionalProperties": False
            },
            "description": "List of key competitors in the space"
        },
        "isBusinessActive": {"type": "boolean", "description": "Whether the business is actively in business and the markdown content is relevant to the name and tagline. Should be false for domain parking, unrelated content, or redirects to unrelated sites."}
    },
    "required": ["details", "valuation", "pros", "concerns", "competitors", "isBusinessActive"],
    "additionalProperties": False
}

# Configure clients
genai.configure(api_key=GEMINI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    return jsonify({
        "business_analysis_prompts": BUSINESS_ANALYSIS_PROMPTS
    })

@app.route('/api/analyze-business', methods=['POST'])
def analyze_business():
    data = request.get_json()
    company_name = data.get('company_name', '')
    company_tagline = data.get('company_tagline', '')
    markdown_content = data.get('markdown_content', '')
    prompt_version = data.get('prompt_version', 'VAMO_V1')
    custom_prompt = data.get('custom_prompt', '')
    
    if not markdown_content:
        return jsonify({"error": "No markdown content provided"}), 400
    
    # Determine which prompt to use
    if prompt_version == 'custom' and custom_prompt:
        system_prompt = custom_prompt
    else:
        system_prompt = BUSINESS_ANALYSIS_PROMPTS.get(prompt_version, BUSINESS_ANALYSIS_PROMPTS["VAMO_V1"])
    
    # Prepare the user content
    user_content = json.dumps({
        "name": company_name,
        "tagline": company_tagline,
        "content": markdown_content
    })
    
    def call_openai_structured(system_prompt, user_content):
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "business_analysis",
                        "strict": True,
                        "schema": BUSINESS_ANALYSIS_SCHEMA
                    }
                },
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_message = data["choices"][0]["message"]["content"]
                try:
                    parsed_analysis = json.loads(assistant_message)
                    return {
                        "model": "OpenAI GPT-4o-mini",
                        "response": parsed_analysis,
                        "raw_response": assistant_message,
                        "error": None
                    }
                except json.JSONDecodeError as e:
                    return {
                        "model": "OpenAI GPT-4o-mini",
                        "response": None,
                        "raw_response": assistant_message,
                        "error": f"JSON Parse Error: {str(e)}"
                    }
            else:
                return {
                    "model": "OpenAI GPT-4o-mini",
                    "response": None,
                    "raw_response": None,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "model": "OpenAI GPT-4o-mini",
                "response": None,
                "raw_response": None,
                "error": str(e)
            }
    
    def call_grok_structured(system_prompt, user_content):
        try:
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Modify the system prompt to explicitly request JSON format
            json_system_prompt = f"{system_prompt}\n\nPlease respond with a valid JSON object that matches this exact schema: {json.dumps(BUSINESS_ANALYSIS_SCHEMA)}\n\nDo not include any text before or after the JSON object."
            
            payload = {
                "model": "grok-2-1212",
                "messages": [
                    {"role": "system", "content": json_system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_message = data["choices"][0]["message"]["content"]
                try:
                    # Clean up the response - remove markdown formatting if present
                    clean_response = assistant_message.strip()
                    if clean_response.startswith('```json'):
                        clean_response = clean_response[7:-3].strip()
                    elif clean_response.startswith('```'):
                        clean_response = clean_response[3:-3].strip()
                    
                    parsed_analysis = json.loads(clean_response)
                    return {
                        "model": "Grok-2-1212",
                        "response": parsed_analysis,
                        "raw_response": assistant_message,
                        "error": None
                    }
                except json.JSONDecodeError as e:
                    return {
                        "model": "Grok-2-1212",
                        "response": None,
                        "raw_response": assistant_message,
                        "error": f"JSON Parse Error: {str(e)} - Raw response: {assistant_message[:200]}..."
                    }
            else:
                return {
                    "model": "Grok-2-1212",
                    "response": None,
                    "raw_response": None,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "model": "Grok-2-1212",
                "response": None,
                "raw_response": None,
                "error": str(e)
            }
    
    def call_gemini_structured(system_prompt, user_content):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            full_prompt = f"{system_prompt}\n\nAnalyze this company data and respond with a valid JSON object matching this schema: {json.dumps(BUSINESS_ANALYSIS_SCHEMA)}\n\nCompany data: {user_content}\n\nJSON Response:"
            
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                )
            )
            
            try:
                # Try to extract JSON from response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()
                
                parsed_analysis = json.loads(response_text)
                return {
                    "model": "Gemini 2.0 Flash",
                    "response": parsed_analysis,
                    "raw_response": response.text,
                    "error": None
                }
            except json.JSONDecodeError as e:
                return {
                    "model": "Gemini 2.0 Flash",
                    "response": None,
                    "raw_response": response.text,
                    "error": f"JSON Parse Error: {str(e)}"
                }
        except Exception as e:
            return {
                "model": "Gemini 2.0 Flash",
                "response": None,
                "raw_response": None,
                "error": str(e)
            }
    
    # Call all models concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_openai = executor.submit(call_openai_structured, system_prompt, user_content)
        future_grok = executor.submit(call_grok_structured, system_prompt, user_content)
        future_gemini = executor.submit(call_gemini_structured, system_prompt, user_content)
        
        results = {
            "openai": future_openai.result(),
            "grok": future_grok.result(),
            "gemini": future_gemini.result()
        }
    
    # Add metadata
    results['prompt_version'] = prompt_version
    results['system_prompt'] = system_prompt
    results['timestamp'] = datetime.now().isoformat()
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3002)
else:
    # For Vercel deployment
    app.debug = False 