import argparse
import sys
from server import start_server

def main():
    parser = argparse.ArgumentParser(description='Image Transfer Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on (default: 5000)')
    
    args = parser.parse_args()
    
    print(f"Starting Image Transfer Server on {args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        start_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
