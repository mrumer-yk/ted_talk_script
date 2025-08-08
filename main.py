# Google Cloud App Engine entry point
import os
from web_app import app

if __name__ == "__main__":
    # For local development
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # For App Engine
    application = app

