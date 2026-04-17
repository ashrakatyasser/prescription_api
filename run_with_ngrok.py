import ngrok
import uvicorn
import os
import threading

os.environ["NGROK_AUTHTOKEN"] = "3CIetF2D29WYYqvGQLZMVJ351uI_63vVDjvZ15Y4zgo6JpBeH"

def run_server():
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)

if __name__ == "__main__":
    # شغّلي السيرفر في thread منفصل
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # شغّلي ngrok
    listener = ngrok.forward(8000, authtoken_from_env=True)
    print(f"\n🌍 Public URL: {listener.url()}\n")

    # خلّيه شغال
    try:
        thread.join()
    except KeyboardInterrupt:
        print("Stopped!")