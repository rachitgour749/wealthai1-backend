#!/usr/bin/env python3
import requests

# Test the current response
response = requests.post('http://localhost:8000/api/chat', json={
    'prompt': 'What is a bond?',
    'system_prompt': 'You are a ChatGPT-style financial expert.'
})

print("Full Response:")
print(response.json().get('response', ''))
print("\n" + "="*50)

# Test with "bond" keyword
response2 = requests.post('http://localhost:8000/api/chat', json={
    'prompt': 'bond',
    'system_prompt': 'You are a ChatGPT-style financial expert.'
})

print("Response for 'bond':")
print(response2.json().get('response', ''))

