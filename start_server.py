#!/usr/bin/env python3
"""
Server startup script with port conflict resolution
"""
import subprocess
import sys
import time
import socket
import os

def is_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """Kill process using the specified port"""
    try:
        # Find process using the port
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    print(f"Killing process {pid} using port {port}")
                    subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True)
                    time.sleep(2)  # Wait for process to terminate
                    return True
        return False
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
        return False

def start_server(port=8000):
    """Start the server on the specified port"""
    print(f"🚀 Starting server on port {port}...")
    
    # Check if port is in use
    if is_port_in_use(port):
        print(f"⚠️  Port {port} is in use. Attempting to free it...")
        if kill_process_on_port(port):
            print(f"✅ Port {port} has been freed")
        else:
            print(f"❌ Could not free port {port}. Trying alternative port...")
            # Try alternative ports
            for alt_port in [8001, 8002, 8003, 8004, 8005]:
                if not is_port_in_use(alt_port):
                    port = alt_port
                    print(f"✅ Using alternative port {port}")
                    break
            else:
                print("❌ No available ports found. Please manually stop processes using ports 8000-8005")
                return False
    
    # Start the server
    try:
        print(f"🌐 Server starting at http://127.0.0.1:{port}")
        subprocess.run([sys.executable, 'server.py'], env={**os.environ, 'PORT': str(port)})
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Get port from command line argument or use default
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port)
