#!/usr/bin/env python3
"""
Test the chatbot API endpoint
"""

import requests
import json

def test_chat_endpoint():
    """Test the chat API endpoint"""
    
    url = 'http://localhost:8000/chat'
    
    # Test with location data included
    test_data = {
        'query': 'What crops should I grow in Punjab?',
        'language': 'en',
        'location': {
            'latitude': 30.3648,
            'longitude': 76.3424,
            'region': 'Punjab'
        }
    }
    
    print('Testing Chat Endpoint with Location Data...')
    print(f'Request: {test_data}')
    print()
    
    try:
        response = requests.post(url, json=test_data, timeout=15)
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print('Chat API Success!')
            print(json.dumps(result, indent=2))
        else:
            print('Chat API Error!')
            print(f'Response: {response.text}')
            
    except requests.exceptions.ConnectionError:
        print('Connection Error - Is the backend server running?')
    except Exception as e:
        print(f'Error: {e}')

def test_chat_without_location():
    """Test chat without location data"""
    
    url = 'http://localhost:8000/chat'
    
    test_data = {
        'query': 'What is the best fertilizer for wheat?',
        'language': 'en'
    }
    
    print('\nTesting Chat Endpoint without Location...')
    print(f'Request: {test_data}')
    print()
    
    try:
        response = requests.post(url, json=test_data, timeout=15)
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print('Chat API Success!')
            print(json.dumps(result, indent=2))
        else:
            print('Chat API Error!')
            print(f'Response: {response.text}')
            
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    test_chat_endpoint()
    test_chat_without_location()