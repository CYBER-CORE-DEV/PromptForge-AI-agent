# PromptForge hosting kit

This project has two deployable parts:
- `backend/`: one Python file that contains the FastAPI server and all three LangGraph agents.
- `frontend/`: one standalone HTML file.

## 1. Deploy the backend on Render

1. Push this project to a GitHub repository. Do not upload a real `.env` file. Keep the `.gitignore` file.
2. In Render, create **New + → Web Service** and connect the GitHub repository.
3. Set **Root Directory** to `backend`.
4. Set **Build Command** to:

   `pip install -r requirements.txt`

5. Set **Start Command** to:

   `uvicorn promptforge_server:app --host 0.0.0.0 --port $PORT`

6. In Render → Environment, add `GROQ_API_KEY` with your real key. Optional: add `MAX_PROMPT_LENGTH=8000` and `MAX_OUTPUT_TOKENS=3000`.
7. Deploy. Your backend URL will look like `https://your-service.onrender.com`.
8. Check `https://your-service.onrender.com/health`. It should return JSON with `status: "ready"` and `api_key_configured: true`.

## 2. Point the frontend at the backend

Open `frontend/index.html` and find:

`const API_BASE = "http://127.0.0.1:8000";`

Replace the URL with the actual HTTPS URL Render assigned to your backend, with no trailing slash, e.g.:

`const API_BASE = "https://your-service.onrender.com";`

Do not put your Groq API key in the HTML.

## 3. Deploy the frontend on Netlify Drop

1. Go to https://app.netlify.com/drop and sign in if requested.
2. Drag the `frontend` folder from this project onto the deploy drop zone.
3. Netlify will publish `index.html` and show a URL such as `https://your-site.netlify.app`.
4. Copy the exact HTTPS origin of that site (just the origin, no path).

## 4. Configure CORS on Render

In Render → your backend service → Environment, add:

`FRONTEND_ORIGINS=https://your-site.netlify.app`

Use your actual Netlify URL. For more than one permitted origin, separate origins with commas. Save and redeploy the backend so the setting is loaded at startup.

The backend currently permits common localhost origins by default only. Adding the hosted frontend origin is required for the browser to call the API from Netlify.

## 5. Verify the full flow

1. Open the Netlify site.
2. Confirm the status label changes to “Backend connected”.
3. Submit a small test prompt.
4. Confirm that the final prompt and the three agent stages appear.
5. If it fails, inspect the Render service logs and browser developer console.

## Security and hosting notes

- Store `GROQ_API_KEY` only in Render Environment Variables, never in frontend JavaScript or Git.
- The frontend URL must use HTTPS when talking to the deployed HTTPS backend.
- `FRONTEND_ORIGINS` must exactly match the hosted frontend origin.
- This server performs three model calls per enhancement. Protect a public deployment from abuse with rate limiting/authentication before sharing it widely.
- Render plan availability and sleep/cold-start behavior can change; check the current plan details before choosing a plan.
