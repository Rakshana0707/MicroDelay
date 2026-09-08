# MicroDelay 🕒

MicroDelay is a personal time estimation intelligence system. It helps users track their morning routines, identify micro-delays (e.g., searching for items, checking phones), and uses machine learning to predict and prevent future lateness.

## Tech Stack
- **Frontend**: React, TypeScript, Vite, Recharts, Lucide Icons
- **Backend**: Python, FastAPI, SQLAlchemy, scikit-learn
- **Database**: SQLite
- **Deployment**: Docker, Docker Compose

## Local Development Setup

### 1. Backend Setup
1. Navigate to the `backend` directory.
2. Create a virtual environment: `python -m venv venv`
3. Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `uvicorn main:app --reload`
   - *Note: The SQLite database (`microdelay.db`) will be created automatically on startup.*
   - *Optional: Run `python seed_demo_users.py` to populate demo accounts.*

### 2. Frontend Setup
1. Navigate to the `frontend` directory.
2. Install dependencies: `npm install`
3. Start the dev server: `npm run dev`

---

## Production Deployment 🚀

This application is configured as a single-container full-stack app for easy deployment. The FastAPI backend is configured to serve the built React frontend static files.

### Deploying with Docker (Recommended)

1. **Build the container:**
   ```bash
   docker-compose build
   ```

2. **Run the container:**
   ```bash
   docker-compose up -d
   ```
   The app will be available at `http://localhost:8000`.

### Deploying to Render, Railway, or Heroku
1. Connect your GitHub repository to the PaaS platform.
2. Select **Docker** as the environment (or specify the `Dockerfile`).
3. Set the following Environment Variables in the platform dashboard:
   - `SECRET_KEY`: A long, random string for JWT security.
   - `ALLOWED_ORIGINS`: E.g., `https://your-domain.com` (or `*` for testing).
4. The platform will automatically build the multi-stage Docker container and expose the FastAPI app on the platform's default port.

### Manual Production Setup
If you prefer not to use Docker:
1. Build the frontend: `cd frontend && npm run build`
2. Configure the backend environment variables (`SECRET_KEY`).
3. Run the backend using a production server like Gunicorn:
   ```bash
   cd backend
   gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
   ```
