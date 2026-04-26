import uvicorn
from pyngrok import ngrok
import sys

def start_ngrok():
    # Set the authtoken provided by the user
    print("[INFO] Configuring ngrok authtoken...")
    ngrok.set_auth_token("3CsqqWd7WsdKf4fubZODjxsX7iK_5TF6zHXYMMzMeNYoekDUc")
    
    # Open a local tunnel to the API port (8000)
    print("[INFO] Starting ngrok tunnel...")
    public_url = ngrok.connect(8000).public_url
    
    print("="*60)
    print(f"SUCCESS! Your API is now live on the internet.")
    print(f"Use this URL in your FlutterFlow app: {public_url}/predict_image")
    print("="*60)

if __name__ == "__main__":
    start_ngrok()
    
    # Start the FastAPI server using Uvicorn
    print("[INFO] Starting local FastAPI server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
