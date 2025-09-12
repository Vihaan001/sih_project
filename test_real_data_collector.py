#!/usr/bin/env python3
"""
Test RealDataCollector methods directly
"""

import sys
import os

# Add the backend directory to Python path
sys.path.append('/c/Users/dpsvi/Documents/crop-recommendation-system/backend')

try:
    from backend.real_data_collector import RealDataCollector
    
    print("Creating RealDataCollector instance...")
    collector = RealDataCollector()
    
    print("\nAvailable methods in RealDataCollector:")
    methods = [method for method in dir(collector) if not method.startswith('_')]
    for method in sorted(methods):
        print(f"  - {method}")
    
    print(f"\nTesting get_comprehensive_data method...")
    if hasattr(collector, 'get_comprehensive_data'):
        print("✅ get_comprehensive_data method EXISTS")
        
        # Test the method
        try:
            result = collector.get_comprehensive_data(23.3441, 85.3096, "Jharkhand")
            print("✅ get_comprehensive_data method WORKS")
            print(f"   Returned keys: {list(result.keys())}")
        except Exception as e:
            print(f"❌ get_comprehensive_data method FAILED: {e}")
    else:
        print("❌ get_comprehensive_data method MISSING")
        
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()