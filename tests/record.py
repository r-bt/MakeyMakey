"""
Records data from the DCA1000
"""
import argparse

def main():
    parser = argparse.ArgumentParser(description='Record data from the DCA1000')
    parser.add_argument('--cfg', type=str, required=True, help='Path to the .lua file')

    args = parser.parse_args()

    
